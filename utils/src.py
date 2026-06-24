#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 16:17:14 2025

@author: Van Tuan NGUYEN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import random
import numpy as np

from . import data_processing, likelihood, statistic, theta_estimation

MAX_HIDDEN_DIM = 6


def set_seed(seed=1):
    random.seed(seed)                            # Python built-in
    np.random.seed(seed)                         # NumPy
    torch.manual_seed(seed)                      # PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Allow warn_only so GPU ops without deterministic kernel don't crash
    torch.use_deterministic_algorithms(True, warn_only=True)

class HIVAE(nn.Module):
    def __init__(self, input_dim, z_dim, s_dim, y_dim, y_dim_partition=[], feat_types_dict=[], intervals_surv_piecewise=None, n_layers_surv_piecewise=2, model_version=None, n_long_outcomes=1):

        super().__init__()
        set_seed()
        self.feat_types_list = feat_types_dict
        self.model_version = model_version
        self._global_norm_params = None  # Phase 1: frozen global normalization
        self.kl_weight_s = 1.0
        self.kl_weight_z = 1.0
        self.feature_group_weights = None

        # Determine Y dimensionality
        if y_dim_partition:
            self.y_dim_partition = y_dim_partition
        else:
            self.y_dim_partition = [y_dim] * len(self.feat_types_list)

        self.x_dim = input_dim
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.y_dim = sum(self.y_dim_partition)

        # for encoder
        self.s_layer = nn.Linear(input_dim, s_dim)  # q(s|x^o)
        self.z_layer = nn.Linear(input_dim + s_dim, z_dim * 2)  # q(z|s,x^o)

        # for decoder
        self.z_distribution_layer = nn.Linear(s_dim, z_dim)  # p(z|s)
        self.y_layer = nn.Linear(z_dim, self.y_dim)  # y deterministic layer
        
        # Store non-module metadata separately (e.g. piecewise intervals)
        self._theta_meta = {}

        theta_modules = {}
        for i, feat in enumerate(self.feat_types_list):

            feat_y_dim = self.y_dim_partition[i]
            key = "feat_" + str(i)

            if feat['type'] in ['real', 'pos']:
                theta_modules[key] = nn.ModuleDict({
                    'mean' : nn.Linear(feat_y_dim + s_dim, 1, bias=False),
                    'sigma' : nn.Linear(s_dim, 1, bias=False),
                })
            elif feat['type'] in ['surv']:
                theta_modules[key] = nn.ModuleDict({
                    'mean_T' : nn.Linear(feat_y_dim + s_dim, 1, bias=False),
                    'sigma_T' : nn.Linear(s_dim, 1, bias=False),
                    'mean_C' : nn.Linear(feat_y_dim + s_dim, 1, bias=False),
                    'sigma_C' : nn.Linear(s_dim, 1, bias=False),
                })

            elif feat['type'] in ['surv_weibull','surv_loglog']:
                theta_modules[key] = nn.ModuleDict({
                    'theta' : nn.Linear(feat_y_dim + s_dim, 4, bias=False),
                })

            elif feat['type'] in ['surv_piecewise']:
                n_intervals = len(intervals_surv_piecewise)
                if n_layers_surv_piecewise == 2:
                    theta_modules[key] = nn.ModuleDict({
                        'theta_T' : nn.Sequential(
                            nn.Linear(feat_y_dim + s_dim, out_features=MAX_HIDDEN_DIM, bias=False),
                            nn.ReLU(),
                            nn.Linear(in_features=MAX_HIDDEN_DIM, out_features=n_intervals, bias=False)
                        ),
                        'theta_C' : nn.Sequential(
                            nn.Linear(feat_y_dim + s_dim, out_features=MAX_HIDDEN_DIM, bias=False),
                            nn.ReLU(),
                            nn.Linear(in_features=MAX_HIDDEN_DIM, out_features=n_intervals, bias=False)
                        ),
                    })
                else:
                    theta_modules[key] = nn.ModuleDict({
                        'theta_T' : nn.Linear(feat_y_dim + s_dim, n_intervals, bias=False),
                        'theta_C' : nn.Linear(feat_y_dim + s_dim, n_intervals, bias=False),
                    })
                self._theta_meta[key] = {'intervals' : intervals_surv_piecewise}

            elif feat['type'] in ['count']:
                theta_modules[key] = nn.Linear(feat_y_dim + s_dim, 1, bias=False)

            elif feat['type'] in ['cat']:
                n_class = int(feat['nclass'])
                theta_modules[key] = nn.Linear(feat_y_dim + s_dim, n_class - 1, bias=False)

            else: # ordinal
                n_class = int(feat['nclass'])
                theta_modules[key] = nn.ModuleDict({
                    'theta' : nn.Linear(s_dim, n_class - 1, bias=False),
                    'mean' : nn.Linear(feat_y_dim + s_dim, 1, bias=False),
                })

        self.theta_layer = nn.ModuleDict(theta_modules)

        # V2A: time-conditioned Gaussian decoder for longitudinal outcomes
        # and lightweight longitudinal summary encoder for the posterior
        if model_version in ('v2a', 'v4_joint', 'v4_seq'):
            self.n_long_outcomes = n_long_outcomes
            time_embed_dim = MAX_HIDDEN_DIM
            self.time_embed_dim = time_embed_dim
            self.time_embed = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim)
            )
            self.longitudinal_mu = nn.Linear(z_dim + s_dim + time_embed_dim, n_long_outcomes, bias=False)
            self.longitudinal_log_var = nn.Linear(s_dim + time_embed_dim, n_long_outcomes, bias=False)

            # Summary encoder: pool observed visits into a fixed-size vector
            # that augments the baseline input for q(s|·) and q(z|·)
            long_summary_dim = MAX_HIDDEN_DIM
            self.long_summary_dim = long_summary_dim
            self.long_summary_net = nn.Sequential(
                nn.Linear(n_long_outcomes + time_embed_dim, long_summary_dim),
                nn.ReLU(),
                nn.Linear(long_summary_dim, long_summary_dim)
            )
            # Re-create encoder layers with augmented input dimension
            augmented_dim = input_dim + long_summary_dim
            self.s_layer = nn.Linear(augmented_dim, s_dim)
            self.z_layer = nn.Linear(augmented_dim + s_dim, z_dim * 2)

        # V4_seq: additional modules for sequential conditional structure
        if model_version == 'v4_seq':
            baseline_summary_dim = MAX_HIDDEN_DIM
            self.baseline_summary_dim = baseline_summary_dim
            # Compute baseline (non-survival) input dimension after encoding
            # (categoricals expand to nclass columns via one-hot)
            baseline_input_dim = 0
            for feat in self.feat_types_list:
                if feat['type'].startswith('surv'):
                    continue
                if feat['type'] in ['cat', 'ordinal']:
                    baseline_input_dim += int(feat.get('nclass', feat.get('dim', 1)))
                else:
                    baseline_input_dim += int(feat.get('dim', 1))
            if baseline_input_dim == 0:
                baseline_input_dim = input_dim  # fallback
            self._baseline_input_dim = baseline_input_dim
            # Baseline summary network: phi_X(X_decoded) -> c_X
            self.baseline_summary_net = nn.Sequential(
                nn.Linear(baseline_input_dim, MAX_HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(MAX_HIDDEN_DIM, baseline_summary_dim)
            )
            # V4_seq longitudinal decoder: augmented with c_X
            self.longitudinal_mu_seq = nn.Linear(
                z_dim + s_dim + time_embed_dim + baseline_summary_dim,
                n_long_outcomes, bias=False
            )
            # Variance does not condition on c_X (same design rationale as V2A)
            # self.longitudinal_log_var is reused from above

            # V4_seq: re-create survival theta layers with augmented input (c_X + r_Y)
            surv_extra_dim = baseline_summary_dim + long_summary_dim
            self._surv_extra_dim = surv_extra_dim
            for i, feat in enumerate(self.feat_types_list):
                key = "feat_" + str(i)
                feat_y_dim = self.y_dim_partition[i]
                if feat['type'] in ['surv_weibull', 'surv_loglog']:
                    self.theta_layer[key] = nn.ModuleDict({
                        'theta': nn.Linear(feat_y_dim + s_dim + surv_extra_dim, 4, bias=False),
                    })
                elif feat['type'] in ['surv_piecewise']:
                    n_intervals = len(intervals_surv_piecewise)
                    if n_layers_surv_piecewise == 2:
                        self.theta_layer[key] = nn.ModuleDict({
                            'theta_T': nn.Sequential(
                                nn.Linear(feat_y_dim + s_dim + surv_extra_dim, MAX_HIDDEN_DIM, bias=False),
                                nn.ReLU(),
                                nn.Linear(MAX_HIDDEN_DIM, n_intervals, bias=False)
                            ),
                            'theta_C': nn.Sequential(
                                nn.Linear(feat_y_dim + s_dim + surv_extra_dim, MAX_HIDDEN_DIM, bias=False),
                                nn.ReLU(),
                                nn.Linear(MAX_HIDDEN_DIM, n_intervals, bias=False)
                            ),
                        })
                    else:
                        self.theta_layer[key] = nn.ModuleDict({
                            'theta_T': nn.Linear(feat_y_dim + s_dim + surv_extra_dim, n_intervals, bias=False),
                            'theta_C': nn.Linear(feat_y_dim + s_dim + surv_extra_dim, n_intervals, bias=False),
                        })
                    self._theta_meta[key] = {'intervals': intervals_surv_piecewise}
                elif feat['type'] in ['surv']:
                    self.theta_layer[key] = nn.ModuleDict({
                        'mean_T': nn.Linear(feat_y_dim + s_dim + surv_extra_dim, 1, bias=False),
                        'sigma_T': nn.Linear(s_dim, 1, bias=False),
                        'mean_C': nn.Linear(feat_y_dim + s_dim + surv_extra_dim, 1, bias=False),
                        'sigma_C': nn.Linear(s_dim, 1, bias=False),
                    })

            # Initialize V4_seq-specific layers with small weights for stability
            with torch.no_grad():
                for layer in [self.baseline_summary_net[0], self.baseline_summary_net[2]]:
                    layer.weight.mul_(0.1)
                    if layer.bias is not None:
                        layer.bias.zero_()
                self.longitudinal_mu_seq.weight.mul_(0.1)
                # Scale down augmented survival theta layers
                for i_f, feat in enumerate(self.feat_types_list):
                    key = "feat_" + str(i_f)
                    if feat['type'].startswith('surv') and key in self.theta_layer:
                        for param_name, param in self.theta_layer[key].named_parameters():
                            param.mul_(0.01)

    def get_theta_view(self):
        """Return a dict-like view of theta_layer that includes non-module metadata (intervals)."""
        theta_view = {}
        for key, mod in self.theta_layer.items():
            if key in self._theta_meta:
                merged = dict(mod)
                merged.update(self._theta_meta[key])
                theta_view[key] = merged
            else:
                theta_view[key] = mod
        return theta_view


    def _encode_longitudinal_summary(self, longitudinal_data):
        """
        Mean-pool observed longitudinal visits into a fixed-size summary vector.

        Parameters
        ----------
        longitudinal_data : tuple (times, values, masks) each (batch, max_visits)
            or values (batch, max_visits, D) for multi-outcome.

        Returns
        -------
        summary : (batch, long_summary_dim)
        """
        times, values, masks = longitudinal_data
        # Per-visit features: [value(s), time_embedding]
        t_embed = self.time_embed(times.unsqueeze(-1))              # (B, V, 16)
        # Ensure values is 3D: (B, V, D)
        if values.dim() == 2:
            values_3d = values.unsqueeze(-1)                        # (B, V, 1)
        else:
            values_3d = values                                      # (B, V, D)
        visit_features = torch.cat([values_3d, t_embed], dim=-1)    # (B, V, D+16)
        visit_encoded = self.long_summary_net(visit_features)       # (B, V, 16)
        # Masked mean pooling over visits
        masks_exp = masks.unsqueeze(-1)                             # (B, V, 1)
        n_visits = masks.sum(dim=1, keepdim=True).clamp(min=1)     # (B, 1)
        summary = (visit_encoded * masks_exp).sum(dim=1) / n_visits # (B, 16)
        return summary


    def forward(
        self,
        batch_data_oberved,
        batch_data,
        batch_miss,
        tau=1.0,
        n_generated_dataset=1,
        longitudinal_data=None,
        encoder_batch_miss=None,
        encoder_l0=None,
    ):
        """
        Forward pass through the encoder and decoder
        """

        # Batch normalization (Phase 1: use frozen global stats if available)
        encoder_miss = batch_miss if encoder_batch_miss is None else encoder_batch_miss
        if self._global_norm_params is not None:
            X_list, normalization_params = data_processing.batch_normalization_frozen(
                batch_data_oberved, self.feat_types_list, encoder_miss,
                self._global_norm_params)
        else:
            X_list, normalization_params = data_processing.batch_normalization(
                batch_data_oberved, self.feat_types_list, encoder_miss)

        if encoder_batch_miss is not None:
            if self._global_norm_params is not None:
                _, normalization_params = data_processing.batch_normalization_frozen(
                    batch_data_oberved, self.feat_types_list, batch_miss,
                    self._global_norm_params)
            else:
                _, normalization_params = data_processing.batch_normalization(
                    batch_data_oberved, self.feat_types_list, batch_miss)

        # Encode
        X = torch.cat(X_list, dim=1)
        static_X = X
        encoder_mask_dim = int(getattr(self, "encoder_mask_dim", 0))
        encoder_l0_dim = int(getattr(self, "encoder_l0_dim", 0))
        if encoder_mask_dim > 0 or encoder_l0_dim > 0:
            if encoder_l0 is None:
                raise ValueError("encoder_l0 is required when the HI-VAE encoder is configured for PhaseSyn L0 augmentation.")
            encoder_l0 = encoder_l0.to(device=X.device, dtype=X.dtype)
            if encoder_l0.shape != (X.shape[0], encoder_l0_dim):
                raise ValueError(f"encoder_l0 must have shape {(X.shape[0], encoder_l0_dim)}, got {tuple(encoder_l0.shape)}.")
            X = torch.cat([X, encoder_miss.to(device=X.device, dtype=X.dtype), encoder_l0], dim=1)

        # V2A / V4: augment encoder input with longitudinal summary
        if self.model_version in ('v2a', 'v4_joint', 'v4_seq'):
            if longitudinal_data is not None:
                long_summary = self._encode_longitudinal_summary(longitudinal_data)
            else:
                long_summary = torch.zeros(X.shape[0], self.long_summary_dim,
                                           device=X.device, dtype=X.dtype)
            X = torch.cat([X, long_summary], dim=1)

        q_params, samples = self.encode(X, tau)

        # V4_seq: compute baseline summary and store for sequential conditioning
        if self.model_version == 'v4_seq':
            # Decode baseline first (standard decode — survival features skipped;
            # they will be decoded below with augmented conditioning)
            p_params, log_p_x, log_p_x_missing, samples = self.decode(
                samples, batch_data, batch_miss, normalization_params,
                n_generated_dataset, skip_surv=True)
            # Compute c_X from reconstructed baseline features
            # Concatenate all non-survival batch_data (already in encoded form)
            X_decoded_parts = []
            for idx_f, feat in enumerate(self.feat_types_list):
                if not feat['type'].startswith('surv'):
                    X_decoded_parts.append(batch_data[idx_f])
            if X_decoded_parts:
                X_decoded = torch.cat(X_decoded_parts, dim=1)
            else:
                X_decoded = torch.zeros(batch_data[0].shape[0], self._baseline_input_dim,
                                       device=batch_data[0].device)
            c_X = self.baseline_summary_net(X_decoded).detach()  # detach to prevent gradient leakage
            samples['c_X'] = c_X

            # Compute longitudinal log-likelihood with c_X conditioning
            longitudinal_loss = torch.tensor(0.0, device=log_p_x.device)
            if longitudinal_data is not None:
                long_log_lik = self.compute_longitudinal_log_lik_seq(samples, longitudinal_data, c_X)
                longitudinal_loss = -torch.mean(long_log_lik)
            else:
                long_log_lik = torch.tensor(0.0, device=log_p_x.device)

            # Compute r_Y for survival conditioning (pre-event visits only)
            if longitudinal_data is not None:
                r_Y = self._compute_pre_event_longitudinal_summary(
                    longitudinal_data, batch_data, batch_miss)
            else:
                r_Y = torch.zeros(samples['z'].shape[0], self.long_summary_dim,
                                  device=samples['z'].device)
            samples['r_Y'] = r_Y

            # Re-compute survival log-likelihood with augmented conditioning
            surv_loss = torch.tensor(0.0, device=log_p_x.device)
            surv_extra = torch.cat([c_X, r_Y], dim=1)
            # Re-decode only survival features with augmented input
            log_p_x_surv, samples = self._decode_survival_v4_seq(
                samples, batch_data, batch_miss, normalization_params,
                n_generated_dataset, surv_extra)
            # Replace survival entries in log_p_x
            for idx_f, feat in enumerate(self.feat_types_list):
                if feat['type'].startswith('surv'):
                    log_p_x[idx_f] = log_p_x_surv[idx_f]

            # Compute loss
            ELBO, loss_reconstruction, KL_z, KL_s = self.loss_function(log_p_x, p_params, q_params)
            # Add longitudinal log-likelihood
            if longitudinal_data is not None:
                ELBO = ELBO + torch.mean(long_log_lik)
        else:
            # Standard decode for V0/V1/V2A/V3/V4_joint
            p_params, log_p_x, log_p_x_missing, samples = self.decode(
                samples, batch_data, batch_miss, normalization_params, n_generated_dataset)

            # Compute loss
            ELBO, loss_reconstruction, KL_z, KL_s = self.loss_function(log_p_x, p_params, q_params)

            # V2A / V4_joint: add longitudinal reconstruction to ELBO
            longitudinal_loss = torch.tensor(0.0, device=ELBO.device)
            if self.model_version in ('v2a', 'v4_joint') and longitudinal_data is not None:
                long_log_lik = self.compute_longitudinal_log_lik(samples, longitudinal_data)
                ELBO = ELBO + torch.mean(long_log_lik)
                longitudinal_loss = -torch.mean(long_log_lik)

        return {
            "samples": samples,
            "log_p_x": log_p_x,
            "log_p_x_missing": log_p_x_missing,
            "loss_re": loss_reconstruction,
            "neg_ELBO_loss": -ELBO,
            "KL_s": KL_s,
            "KL_z": KL_z,
            "p_params": p_params,
            "q_params": q_params,
            "longitudinal_loss": longitudinal_loss,
            "encoder_input": X,
            "encoder_static_input": static_X,
        }

    def decode(self, samples, batch_data_list, miss_list, normalization_params, n_generated_dataset=1, skip_surv=False):
        """
        Decodes latent variables into output reconstructions.

        Parameters:
        -----------
        samples : dict
            Sampled latent variables {s, z}.

        batch_data_list : list of torch.Tensor
            Original batch data.

        miss_list : torch.Tensor
            Mask indicating missing data.

        normalization_params : dict
            Normalization parameters for data.

        skip_surv : bool
            If True, skip survival features during decoding (used by V4_seq).

        Returns:
        --------
        
        p_params : dict
            Parameters of the prior distributions.
        
        log_p_x : torch.Tensor
            Log-likelihood of observed data.
        
        log_p_x_missing : torch.Tensor
            Log-likelihood of missing data.

        samples : dict
            Updated dictionary containing generated samples.
        """
        p_params = {}

        # Compute p(z|s)
        mean_pz, log_var_pz = statistic.z_prior_GMM(samples["s"], self.z_distribution_layer)
        p_params["z"] = (mean_pz, log_var_pz)

        # Compute deterministic y layer
        samples["y"] = self.y_layer(samples["z"])

        # Partition y
        grouped_samples_y = data_processing.y_partition(samples["y"], self.feat_types_list, self.y_dim_partition)

        if skip_surv:
            # V4_seq: decode only non-survival features; survival will be decoded
            # separately with augmented conditioning in _decode_survival_v4_seq
            non_surv_indices = [i for i, f in enumerate(self.feat_types_list)
                                if not f['type'].startswith('surv')]
            surv_indices = [i for i, f in enumerate(self.feat_types_list)
                            if f['type'].startswith('surv')]

            if non_surv_indices:
                sub_y = [grouped_samples_y[i] for i in non_surv_indices]
                sub_feat = [self.feat_types_list[i] for i in non_surv_indices]
                sub_miss = miss_list[:, non_surv_indices]
                sub_data = [batch_data_list[i] for i in non_surv_indices]
                sub_norm = [normalization_params[i] for i in non_surv_indices]
                # Build sub theta_view
                sub_theta_view = {}
                for j_new, j_orig in enumerate(non_surv_indices):
                    orig_key = "feat_" + str(j_orig)
                    new_key = "feat_" + str(j_new)
                    tv = self.get_theta_view()
                    if orig_key in tv:
                        sub_theta_view[new_key] = tv[orig_key]

                sub_theta = theta_estimation.theta_estimation_from_ys(
                    sub_y, samples["s"], sub_feat, sub_miss, sub_theta_view)
                _, sub_log_p_x, sub_log_p_x_miss, sub_samples_x = \
                    likelihood.loglik_evaluation(
                        sub_data, sub_feat, sub_miss, sub_theta, sub_norm,
                        n_generated_dataset)
            else:
                sub_log_p_x = torch.zeros(0, batch_data_list[0].shape[0],
                                          device=samples["z"].device)
                sub_log_p_x_miss = sub_log_p_x
                sub_samples_x = []

            # Reconstruct full-size outputs with zeros for survival slots
            n_feat = len(self.feat_types_list)
            batch_size = samples["z"].shape[0]
            log_p_x_full = []
            log_p_x_miss_full = []
            samples_x_full = [None] * n_feat
            params_x_full = [None] * n_feat
            j_sub = 0
            for i in range(n_feat):
                if i in surv_indices:
                    log_p_x_full.append(torch.zeros(batch_size, device=samples["z"].device))
                    log_p_x_miss_full.append(torch.zeros(batch_size, device=samples["z"].device))
                else:
                    log_p_x_full.append(sub_log_p_x[j_sub])
                    log_p_x_miss_full.append(sub_log_p_x_miss[j_sub])
                    if j_sub < len(sub_samples_x):
                        samples_x_full[i] = sub_samples_x[j_sub]
                    j_sub += 1
            log_p_x = torch.stack(log_p_x_full)
            log_p_x_missing = torch.stack(log_p_x_miss_full)
            samples["x"] = samples_x_full
            p_params["x"] = params_x_full
        else:
            # Standard full decode
            # Compute θ parameters
            theta = theta_estimation.theta_estimation_from_ys(grouped_samples_y, samples["s"], self.feat_types_list, miss_list, self.get_theta_view())

            # Compute log-likelihood and reconstructed data
            p_params["x"], log_p_x, log_p_x_missing, samples["x"] = likelihood.loglik_evaluation(
                batch_data_list, self.feat_types_list, miss_list, theta, normalization_params, n_generated_dataset
            )
        return p_params, log_p_x, log_p_x_missing, samples


    def loss_function(self, log_p_x, p_params, q_params):
        """
        Computes the Evidence Lower Bound (ELBO) for the Variational Autoencoder.

        Parameters:
        -----------
        log_p_x : torch.Tensor
            Log-likelihood of reconstructed samples.
        
        p_params : dict
            Parameters of prior distributions.
        
        q_params : dict
            Parameters of variational distributions.

        Returns:
        --------
        ELBO : torch.Tensor
            Evidence Lower Bound loss.
        
        loss_reconstruction : torch.Tensor
            Reconstruction loss term.
        
        KL_z : torch.Tensor
            KL divergence for z.
        
        KL_s : torch.Tensor
            KL divergence for s.
        """

        # KL(q(s|x) || p(s))
        log_pi = q_params['s']
        pi_param = F.softmax(log_pi, dim=-1)
        # KL_s is the batch mean of per-subject categorical KL terms, which is
        # equivalent to placing KL_s inside the minibatch sum for the scalar ELBO.
        KL_s = -F.cross_entropy(log_pi, pi_param, reduction='mean') + torch.log(torch.tensor(float(self.s_dim), device=log_pi.device))

        # KL(q(z|s,x) || p(z|s))
        mean_pz, log_var_pz = p_params['z']
        mean_qz, log_var_qz = q_params['z']
        
        # KL_z is per subject conditional on the sampled/relaxed s.
        KL_z = -0.5 * self.z_dim + 0.5 * torch.sum(
            torch.exp(log_var_qz - log_var_pz) + (mean_pz - mean_qz).pow(2) / torch.exp(log_var_pz) - log_var_qz + log_var_pz, dim=1
        )
        # Expectation of log p(x|y)
        feature_weights = getattr(self, "feature_group_weights", None)
        if feature_weights is not None:
            feature_weights = feature_weights.to(log_p_x.device, dtype=log_p_x.dtype).view(-1, 1)
            log_p_x = log_p_x * feature_weights
        loss_reconstruction = torch.sum(log_p_x, dim=0)

        # Complete ELBO
        kl_weight_z = float(getattr(self, "kl_weight_z", 1.0))
        kl_weight_s = float(getattr(self, "kl_weight_s", 1.0))
        ELBO = torch.mean(loss_reconstruction - kl_weight_z * KL_z - kl_weight_s * KL_s, dim=0)

        return ELBO, loss_reconstruction, KL_z, KL_s


    # ------------------------------------------------------------------
    # V2A: longitudinal decoder helpers
    # ------------------------------------------------------------------

    def compute_longitudinal_log_lik(self, samples, longitudinal_data):
        """
        Masked Gaussian log-likelihood for V2A repeated outcomes.

        Parameters
        ----------
        samples : dict with 'z' (batch, z_dim) and 's' (batch, s_dim).
        longitudinal_data : tuple (times, values, masks).
            times: (batch, max_visits)
            values: (batch, max_visits) or (batch, max_visits, D)
            masks: (batch, max_visits)

        Returns
        -------
        log_lik : (batch,) per-patient masked log-likelihood.
        """
        times, values, masks = longitudinal_data
        batch_size, max_visits = times.shape
        z = samples["z"]
        s = samples["s"]

        t_embed = self.time_embed(times.unsqueeze(-1))          # (B, V, D_t)
        z_exp = z.unsqueeze(1).expand(-1, max_visits, -1)       # (B, V, D_z)
        s_exp = s.unsqueeze(1).expand(-1, max_visits, -1)       # (B, V, D_s)

        mu = self.longitudinal_mu(
            torch.cat([z_exp, s_exp, t_embed], dim=-1)
        )                                                       # (B, V, D)
        log_var_raw = self.longitudinal_log_var(
            torch.cat([s_exp, t_embed], dim=-1)
        )                                                       # (B, V, D)
        var = F.softplus(log_var_raw).clamp(min=1e-3, max=1e3)

        # Ensure values is 3D: (B, V, D)
        if values.dim() == 2:
            values = values.unsqueeze(-1)                       # (B, V, 1)

        nll_per_visit = 0.5 * (torch.log(var)
                                + (values - mu) ** 2 / var
                                + torch.log(torch.tensor(2.0 * torch.pi, device=var.device)))
        # Expand mask to (B, V, D) for broadcasting
        masks_exp = masks.unsqueeze(-1).expand_as(nll_per_visit)
        log_lik = -(nll_per_visit * masks_exp).sum(dim=(1, 2))  # (B,)
        return log_lik

    def generate_longitudinal(self, samples, time_grid, n_samples=1):
        """
        Sample longitudinal trajectories on a normalised time grid.

        Parameters
        ----------
        samples : dict with 'z' and 's'.
        time_grid : (n_times,) tensor of normalised times.
        n_samples : int – number of trajectory draws per patient.

        Returns
        -------
        mu       : (batch, n_times, D)
        var      : (batch, n_times, D)
        generated: (n_samples, batch, n_times, D)
        """
        z, s = samples["z"], samples["s"]
        batch_size = z.shape[0]
        n_times = time_grid.shape[0]

        times = time_grid.unsqueeze(0).expand(batch_size, -1)
        t_embed = self.time_embed(times.unsqueeze(-1))
        z_exp = z.unsqueeze(1).expand(-1, n_times, -1)
        s_exp = s.unsqueeze(1).expand(-1, n_times, -1)

        mu = self.longitudinal_mu(
            torch.cat([z_exp, s_exp, t_embed], dim=-1)
        )                                                       # (B, T, D)
        log_var_raw = self.longitudinal_log_var(
            torch.cat([s_exp, t_embed], dim=-1)
        )                                                       # (B, T, D)
        var = F.softplus(log_var_raw).clamp(min=1e-3, max=1e3)

        generated = torch.distributions.Normal(mu, torch.sqrt(var)).sample((n_samples,))
        return mu, var, generated

    # ------------------------------------------------------------------
    # V4_seq: sequential conditional helpers
    # ------------------------------------------------------------------

    def compute_longitudinal_log_lik_seq(self, samples, longitudinal_data, c_X):
        """
        V4_seq longitudinal log-likelihood conditioned on baseline summary c_X.

        Uses longitudinal_mu_seq instead of longitudinal_mu.
        """
        times, values, masks = longitudinal_data
        batch_size, max_visits = times.shape
        z = samples["z"]
        s = samples["s"]

        t_embed = self.time_embed(times.unsqueeze(-1))
        z_exp = z.unsqueeze(1).expand(-1, max_visits, -1)
        s_exp = s.unsqueeze(1).expand(-1, max_visits, -1)
        c_X_exp = c_X.unsqueeze(1).expand(-1, max_visits, -1)

        mu = self.longitudinal_mu_seq(
            torch.cat([z_exp, s_exp, t_embed, c_X_exp], dim=-1)
        )
        log_var_raw = self.longitudinal_log_var(
            torch.cat([s_exp, t_embed], dim=-1)
        )
        var = F.softplus(log_var_raw).clamp(min=1e-3, max=1e3)

        if values.dim() == 2:
            values = values.unsqueeze(-1)

        nll_per_visit = 0.5 * (torch.log(var)
                                + (values - mu) ** 2 / var
                                + torch.log(torch.tensor(2.0 * torch.pi, device=var.device)))
        masks_exp = masks.unsqueeze(-1).expand_as(nll_per_visit)
        log_lik = -(nll_per_visit * masks_exp).sum(dim=(1, 2))
        return log_lik

    def _compute_pre_event_longitudinal_summary(self, longitudinal_data, batch_data, batch_miss):
        """
        Compute longitudinal summary r_Y using only pre-event visits.

        This prevents information leakage: the survival head should not
        see longitudinal observations that occur after the event/censoring time.
        """
        times, values, masks = longitudinal_data
        batch_size = times.shape[0]

        # Find the survival time for each patient from batch_data
        surv_time = None
        for idx_f, feat in enumerate(self.feat_types_list):
            if feat['type'].startswith('surv'):
                surv_time = batch_data[idx_f][:, 0]  # first col is time
                break

        if surv_time is not None:
            # Mask visits after the event time (pre-event only)
            pre_event_mask = (times <= surv_time.unsqueeze(1)).float() * masks
        else:
            pre_event_mask = masks

        pre_event_long_data = (times, values, pre_event_mask)
        return self._encode_longitudinal_summary(pre_event_long_data)

    def _decode_survival_v4_seq(self, samples, batch_data_list, miss_list,
                                 normalization_params, n_generated_dataset, surv_extra):
        """
        Re-decode only survival features with augmented conditioning (c_X, r_Y).

        For V4_seq, survival theta layers expect input [y^(j); s; c_X; r_Y].
        """
        grouped_samples_y = data_processing.y_partition(
            samples["y"], self.feat_types_list, self.y_dim_partition)

        log_p_x_list = []
        for i, feat in enumerate(self.feat_types_list):
            if not feat['type'].startswith('surv'):
                # Use a placeholder — these were already computed
                log_p_x_list.append(None)
                continue

            key = "feat_" + str(i)
            y_i = grouped_samples_y[i]
            s = samples["s"]
            mask = miss_list[:, i].bool()

            # Augmented input: [y; s; surv_extra]
            observed_y, missing_y = y_i[mask], y_i[~mask]
            observed_s, missing_s = s[mask], s[~mask]
            observed_extra, missing_extra = surv_extra[mask], surv_extra[~mask]
            condition_indices = [~mask, mask]

            # Compute theta with augmented inputs
            theta_layer_i = self.get_theta_view()[key]
            if feat['type'] in ['surv_weibull', 'surv_loglog']:
                h_out = theta_estimation.observed_data_layer(
                    torch.cat([observed_y, observed_s, observed_extra], dim=1),
                    torch.cat([missing_y, missing_s, missing_extra], dim=1),
                    condition_indices,
                    layer=theta_layer_i["theta"]).T
                theta_i = list(h_out)
            elif feat['type'] == 'surv_piecewise':
                h_T = theta_estimation.observed_data_layer(
                    torch.cat([observed_y, observed_s, observed_extra], dim=1),
                    torch.cat([missing_y, missing_s, missing_extra], dim=1),
                    condition_indices,
                    layer=theta_layer_i["theta_T"])
                h_C = theta_estimation.observed_data_layer(
                    torch.cat([observed_y, observed_s, observed_extra], dim=1),
                    torch.cat([missing_y, missing_s, missing_extra], dim=1),
                    condition_indices,
                    layer=theta_layer_i["theta_C"])
                theta_i = [h_T, h_C, theta_layer_i["intervals"]]
            elif feat['type'] == 'surv':
                h_mean_T = theta_estimation.observed_data_layer(
                    torch.cat([observed_y, observed_s, observed_extra], dim=1),
                    torch.cat([missing_y, missing_s, missing_extra], dim=1),
                    condition_indices,
                    layer=theta_layer_i["mean_T"])
                h_sigma_T = theta_estimation.observed_data_layer(
                    observed_s, missing_s, condition_indices,
                    layer=theta_layer_i["sigma_T"])
                h_mean_C = theta_estimation.observed_data_layer(
                    torch.cat([observed_y, observed_s, observed_extra], dim=1),
                    torch.cat([missing_y, missing_s, missing_extra], dim=1),
                    condition_indices,
                    layer=theta_layer_i["mean_C"])
                h_sigma_C = theta_estimation.observed_data_layer(
                    observed_s, missing_s, condition_indices,
                    layer=theta_layer_i["sigma_C"])
                theta_i = [h_mean_T, h_sigma_T, h_mean_C, h_sigma_C]
            else:
                theta_i = None

            if theta_i is not None:
                loglik_fn = getattr(likelihood, 'loglik_' + feat['type'])
                batch_data_ext = [batch_data_list[i], miss_list[:, i]]
                out = loglik_fn(batch_data_ext, feat, theta_i,
                               normalization_params[i], n_generated_dataset)
                log_p_x_list.append(out['log_p_x'])
                # Update generated survival samples
                if 'x' in samples:
                    samples['x'][i] = out['samples']
            else:
                log_p_x_list.append(None)

        # Build log_p_x tensor — fill survival entries
        n_feats = len(self.feat_types_list)
        batch_size = samples['z'].shape[0]
        log_p_x_out = [None] * n_feats
        for i in range(n_feats):
            if log_p_x_list[i] is not None:
                log_p_x_out[i] = log_p_x_list[i]
            else:
                log_p_x_out[i] = torch.zeros(batch_size, device=samples['z'].device)
        return torch.stack(log_p_x_out), samples

    def generate_longitudinal_seq(self, samples, time_grid, c_X, n_samples=1):
        """
        V4_seq: generate longitudinal trajectories conditioned on baseline summary c_X.
        """
        z, s = samples["z"], samples["s"]
        batch_size = z.shape[0]
        n_times = time_grid.shape[0]

        times = time_grid.unsqueeze(0).expand(batch_size, -1)
        t_embed = self.time_embed(times.unsqueeze(-1))
        z_exp = z.unsqueeze(1).expand(-1, n_times, -1)
        s_exp = s.unsqueeze(1).expand(-1, n_times, -1)
        c_X_exp = c_X.unsqueeze(1).expand(-1, n_times, -1)

        mu = self.longitudinal_mu_seq(
            torch.cat([z_exp, s_exp, t_embed, c_X_exp], dim=-1)
        )
        log_var_raw = self.longitudinal_log_var(
            torch.cat([s_exp, t_embed], dim=-1)
        )
        var = F.softplus(log_var_raw).clamp(min=1e-3, max=1e3)

        generated = torch.distributions.Normal(mu, torch.sqrt(var)).sample((n_samples,))
        return mu, var, generated

    # ------------------------------------------------------------------
    # Phase 1: Post-generation longitudinal truncation
    # ------------------------------------------------------------------

    @staticmethod
    def truncate_longitudinal_at_event(trajectories, time_grid, event_times,
                                        return_planned=False):
        """Truncate/mask generated longitudinal visits after sampled event time.

        Parameters
        ----------
        trajectories : (n_samples, batch, n_times, D) or (batch, n_times, D)
            Generated trajectories.
        time_grid : (n_times,) tensor of normalised times.
        event_times : (batch,) observed/sampled event times (normalised).
        return_planned : bool
            If True, return both planned and observed trajectories.

        Returns
        -------
        observed : same shape as trajectories, with post-event visits zeroed out.
        mask : (batch, n_times) binary mask (1 = pre/at event, 0 = post-event).
        planned : (optional) original trajectories, only if return_planned=True.
        """
        # Expand time_grid to match batch dimension
        # event_times: (batch,), time_grid: (n_times,)
        mask = (time_grid.unsqueeze(0) <= event_times.unsqueeze(1)).float()  # (batch, n_times)

        if trajectories.dim() == 4:
            # (n_samples, batch, n_times, D)
            mask_exp = mask.unsqueeze(0).unsqueeze(-1)  # (1, batch, n_times, 1)
            observed = trajectories * mask_exp
        elif trajectories.dim() == 3:
            # (batch, n_times, D)
            mask_exp = mask.unsqueeze(-1)  # (batch, n_times, 1)
            observed = trajectories * mask_exp
        else:
            observed = trajectories

        if return_planned:
            return observed, mask, trajectories
        return observed, mask

    # ------------------------------------------------------------------
    # Phase 2B: Longitudinal summary factory
    # ------------------------------------------------------------------

    def _build_long_summary_net(self, n_long_outcomes, time_embed_dim,
                                 summary_type="mean_pool_16"):
        """Factory for longitudinal summary encoder variants.

        Parameters
        ----------
        summary_type : str
            One of: 'mean_pool_16' (default), 'mean_pool_32', 'mean_pool_64',
            'gru_pool', 'attention_pool'.
        """
        if summary_type == "mean_pool_16":
            dim = MAX_HIDDEN_DIM
            net = nn.Sequential(
                nn.Linear(n_long_outcomes + time_embed_dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
            return net, dim
        elif summary_type == "mean_pool_32":
            dim = MAX_HIDDEN_DIM
            net = nn.Sequential(
                nn.Linear(n_long_outcomes + time_embed_dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
            return net, dim
        elif summary_type == "mean_pool_64":
            dim = MAX_HIDDEN_DIM
            net = nn.Sequential(
                nn.Linear(n_long_outcomes + time_embed_dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
            return net, dim
        elif summary_type == "gru_pool":
            dim = MAX_HIDDEN_DIM
            net = nn.GRU(n_long_outcomes + time_embed_dim, dim, batch_first=True)
            return net, dim
        elif summary_type == "attention_pool":
            dim = MAX_HIDDEN_DIM
            net = nn.Sequential(
                nn.Linear(n_long_outcomes + time_embed_dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
            # Attention weight network
            self._attn_weight_net = nn.Sequential(
                nn.Linear(dim, 1),
                nn.Softmax(dim=1)
            )
            return net, dim
        else:
            raise ValueError(f"Unknown summary_type: {summary_type}")

    # ------------------------------------------------------------------
    # Phase 2C: Survival-aware encoder embedding (ablation, disabled by default)
    # ------------------------------------------------------------------

    def _build_survival_embedding(self, embed_dim=MAX_HIDDEN_DIM):
        """Build optional survival embedding for encoder-side ablation.

        Input features: [log1p(t), delta, delta*log1p(t), (1-delta)*log1p(t)]
        Output: embed_dim-dimensional learned embedding.
        """
        embed_dim = min(int(embed_dim), MAX_HIDDEN_DIM)
        return nn.Sequential(
            nn.Linear(4, MAX_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(MAX_HIDDEN_DIM, embed_dim)
        )

    def compute_survival_embedding(self, batch_data):
        """Compute censoring-aware survival embedding from batch data.

        Only used when surv_encoder_ablation is enabled.
        """
        if not hasattr(self, 'surv_embed_net'):
            return None
        # Find survival feature data
        for idx_f, feat in enumerate(self.feat_types_list):
            if feat['type'].startswith('surv'):
                surv_data = batch_data[idx_f]
                t = surv_data[:, 0]
                delta = surv_data[:, 1]
                log_t = torch.log1p(t)
                features = torch.stack([
                    log_t, delta,
                    delta * log_t,
                    (1 - delta) * log_t
                ], dim=1)  # (batch, 4)
                return self.surv_embed_net(features)
        return None


class HIVAE_factorized(HIVAE):

    """
    The HI_VAE Model with factorized encoder.

    This model encodes input data into latent variables (s, z) using variational inference,
    and decodes these representations to reconstruct the original input.

    Parameters:
    -----------
    input_dim : int
        Dimensionality of input data.
    
    z_dim : int
        Dimensionality of the latent variable z.
    
    s_dim : int
        Number of categorical latent states (s).

    y_dim : int
        Dimensionality of the deterministic layer y.

    y_dim_partition : list
        Partitioning dimensions for input variables.

    feat_types_file : str
        
    """

    def __init__(self, input_dim, z_dim, s_dim, y_dim, y_dim_partition, feat_types_dict, intervals_surv_piecewise, n_layers_surv_piecewise=2, model_version=None, n_long_outcomes=1):

        # print(f'[*] Importing model: {model_name}')
        super().__init__(input_dim, z_dim, s_dim, y_dim, y_dim_partition, feat_types_dict, intervals_surv_piecewise, n_layers_surv_piecewise, model_version, n_long_outcomes)

    def encode(self, X, tau):
        """
        Encodes input data X into latent variables s and z using variational inference.

        Parameters:
        -----------
        X : torch.Tensor
            Input data batch.
        tau : float
            Temperature parameter for Gumbel-softmax.

        Returns:
        --------
        q_params : dict
            Parameters of the variational distributions {s_logits, (mean_qz, log_var_qz)}.

        samples : dict
            Sampled latent variables {s, z}.
        """

        # Softmax over s (categorical distribution)
        logits_s = self.s_layer(X)
        p_s = F.softmax(logits_s, dim=-1)

        # Gumbel-softmax trick
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(p_s)))
        samples_s = F.softmax(torch.log(torch.clamp(p_s, 1e-6, 1)) + gumbel_noise / tau, dim=-1)

        # Compute q(z|s,x^o)
        z_params = self.z_layer(torch.cat([X, samples_s], dim=1))
        mean_qz, log_var_qz = torch.chunk(z_params, 2, dim=1)
        log_var_qz = torch.clamp(log_var_qz, -15.0, 15.0)

        # Reparametrization trick
        eps = torch.randn_like(mean_qz)
        samples_z = mean_qz + torch.exp(log_var_qz / 2) * eps

        q_params = {"s": logits_s, "z": (mean_qz, log_var_qz)}
        samples = {"s": samples_s, "z": samples_z}

        return q_params, samples



class HIVAE_inputDropout(HIVAE):

    """
    The HI_VAE model with input dropout encoder.

    This model encodes input data into latent variables (s, z) using variational inference,
    and decodes these representations to reconstruct the original input.

    Parameters:
    -----------
    input_dim : int
        Dimensionality of input data.

    z_dim : int
        Dimensionality of the latent variable z.

    s_dim : int
        Number of categorical latent states (s).

    y_dim : int
        Dimensionality of the deterministic layer y.

    y_dim_partition : list
        Partitioning dimensions for input variables.

    feat_types_file : str

    """

    def __init__(self, input_dim, z_dim, s_dim, y_dim, y_dim_partition, feat_types_dict, intervals_surv_piecewise, n_layers_surv_piecewise=2, model_version=None, n_long_outcomes=1):

        # print(f'[*] Importing model: {model_name}')
        super().__init__(input_dim, z_dim, s_dim, y_dim, y_dim_partition, feat_types_dict, intervals_surv_piecewise, n_layers_surv_piecewise, model_version, n_long_outcomes)
    
    def encode(self, X, tau):
        """
        Encodes input data X into latent variables s and z using variational inference.

        Parameters:
        -----------
        X : torch.Tensor
            Input data batch.
        tau : float
            Temperature parameter for Gumbel-softmax.

        Returns:
        --------        
        q_params : dict
            Parameters of the variational distributions {s_logits, (mean_qz, log_var_qz)}.

        samples : dict
            Sampled latent variables {s, z}.
        """

        #Create the proposal of q(s|x^o)
        samples_s, s_params = statistic.s_proposal_multinomial(X, self.s_layer, tau)

        # Compute q(z|s,x^o)
        batch_size = X.shape[0]
        samples_z, z_params = statistic.z_proposal_GMM(X, samples_s, batch_size, self.z_dim, self.z_layer)

        q_params = {"s": s_params, "z": z_params} 
        samples = {"s": samples_s, "z": samples_z}

        return q_params, samples
