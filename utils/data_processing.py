#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 20:35:11 2025

@author: Van Tuan NGUYEN
"""

import argparse
import os
import csv
import numpy as np
import torch
import pandas as pd


def get_args(argv = None):
    parser = argparse.ArgumentParser(description='Default parameters of the models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=200, help='Size of the batches')
    parser.add_argument('--epochs',type=int,default=5001, help='Number of epochs of the simulations')
    parser.add_argument('--perp',type=int,default=10, help='Perplexity for the t-SNE')
    parser.add_argument('--train', type=int,default=1, help='Training model flag')
    parser.add_argument('--display', type=int,default=1, help='Display option flag')
    parser.add_argument('--save', type=int,default=1000, help='Save variables every save iterations')
    parser.add_argument('--restore', type=int,default=0, help='To restore session, to keep training or evaluation') 
    parser.add_argument('--plot', type=int,default=1, help='Plot results flag')
    parser.add_argument('--dim_latent_s',type=int,default=10, help='Dimension of the categorical space')
    parser.add_argument('--dim_latent_z',type=int,default=2, help='Dimension of the Z latent space')
    parser.add_argument('--dim_latent_y',type=int,default=10, help='Dimension of the Y latent space')
    parser.add_argument('--dim_latent_y_partition',type=int, nargs='+', help='Partition of the Y latent space')
    parser.add_argument('--miss_percentage_train',type=float,default=0.0, help='Percentage of missing data in training')
    parser.add_argument('--miss_percentage_test',type=float,default=0.0, help='Percentage of missing data in test')
    parser.add_argument('--model_name', type=str, default='model_new', help='File of the training model')
    parser.add_argument('--save_file', type=str, default='new_mnist_zdim5_ydim10_4images_', help='Save file name')
    parser.add_argument('--data_file', type=str, default='MNIST_data', help='File with the data')
    parser.add_argument('--types_file', type=str, default='mnist_train_types2.csv', help='File with the types of the data')
    parser.add_argument('--miss_file', type=str, default='Missing_test.csv', help='File with the missing indexes mask')
    parser.add_argument('--true_miss_file', type=str, help='File with the missing indexes when there are NaN in the data')

    # Multi-version support
    parser.add_argument('--model_version', type=str, default=None,
                        choices=['v0', 'v1', 'v2a', 'phasesyn_weibull', 'phasesyn_piecewise'],
                        help='Model version: v0=baseline, v1=baseline+endpoint, v2a=longitudinal, phasesyn_*=survival')
    parser.add_argument('--endpoint_column', type=str, default=None,
                        help='Endpoint column name (for V1)')
    parser.add_argument('--longitudinal_file', type=str, default=None,
                        help='CSV file with longitudinal data in long format (for V2A)')
    parser.add_argument('--patient_id_col', type=str, default='patient_id',
                        help='Patient ID column name in longitudinal file (for V2A)')
    parser.add_argument('--time_col', type=str, default='visit_time',
                        help='Visit time column name in longitudinal file (for V2A)')
    parser.add_argument('--longitudinal_value_col', type=str, default='value',
                        help='Longitudinal outcome value column (for V2A)')
    parser.add_argument('--longitudinal_mask_col', type=str, default=None,
                        help='Optional observation mask column in longitudinal file (for V2A)')
    parser.add_argument('--time_grid', type=float, nargs='+', default=None,
                        help='Time grid for longitudinal generation (for V2A)')
    parser.add_argument('--use_controls_only', action='store_true', default=False,
                        help='Use control group only')

    return parser.parse_args(argv)

def read_data(data_file, types_file, miss_file, true_miss_file, surv_type=None):
    """
    Reads data from CSV files, handles missing values, and applies necessary transformations.

    Parameters:
    -----------
    data_file : str
        Path to the CSV file containing the dataset.
    
    types_file : str
        Path to the CSV file specifying the data types and dimensions for each feature.
    
    miss_file : str
        Path to the CSV file indicating the missing values in the dataset.
    
    true_miss_file : str or None
        Path to the CSV file containing the true missing value mask, if available.

    surv_type : str, default=None
        Type identifier for the survival outcome.

    Returns:
    --------
    data : torch.Tensor
        Transformed dataset with categorical, ordinal, and continuous values properly encoded.
    
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.
    
    miss_mask : torch.Tensor
        A binary mask indicating observed (1) and missing (0) values.
    
    true_miss_mask : torch.Tensor
        A binary mask indicating the actual missing values, if provided.
    
    n_samples : int
        The number of samples in the dataset.
    """
    
    # Read types of data from types file
    with open(types_file) as f:
        types_dict = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
    if surv_type is not None:
        for i in range(len(types_dict)):
            if types_dict[i]["name"] == "survcens":
                types_dict[i]["type"] = surv_type

    # Read data from input file and convert to PyTorch tensor
    with open(data_file, 'r') as f:
        data = [[float(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
        data = torch.tensor(data, dtype=torch.float32)
    
    # Handle true missing values if provided
    if true_miss_file:
        with open(true_miss_file, 'r') as f:
            missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            missing_positions = torch.tensor(missing_positions, dtype=torch.long)

        true_miss_mask = torch.ones((data.shape[0], len(types_dict)))
        true_miss_mask[missing_positions[:, 0] - 1, missing_positions[:, 1] - 1] = 0  # CSV indexes start at 1
        
        # Replace NaNs with appropriate default values
        nan_mask = torch.isnan(data)
        data_filler = torch.zeros(data.shape[1], dtype=torch.float32)
        
        for i, dtype in enumerate(types_dict):
            if dtype['type'] in {'cat', 'ordinal'}:
                unique_vals = torch.unique(data[:, i][~nan_mask[:, i]])  # Get unique non-NaN values
                data_filler[i] = unique_vals[0] if len(unique_vals) > 0 else 0  # Fill with first category
            else:
                data_filler[i] = 0.0  # Fill numerical data with 0
        
        data[nan_mask] = data_filler.repeat(data.shape[0], 1)[nan_mask]
    
    else:
        true_miss_mask = torch.ones((data.shape[0], len(types_dict)))  # No effect on data if no file is provided
    
    # Construct processed data matrices
    data_complete = []
    
    feat_idx = 0
    feat_names = []
    for i, feature in enumerate(types_dict):

        if feature['type'] == 'cat':
            # One-hot encoding for categorical data
            cat_data = data[:, feat_idx].to(torch.int64)
            unique_vals, indexes = torch.unique(cat_data, return_inverse=True)
            new_categories = torch.arange(int(feature['nclass']), dtype=torch.int64)
            mapped_categories = new_categories[indexes]
            
            one_hot = torch.zeros((data.shape[0], len(new_categories)))
            one_hot[torch.arange(data.shape[0]), mapped_categories] = 1
            data_complete.append(one_hot)
            feat_names += [feature['name'] + "_" + str(j) for j in np.arange(len(new_categories))]
        
        elif feature['type'] == 'ordinal':
            # Thermometer encoding for ordinal data
            ordinal_data = data[:, feat_idx].to(torch.int64)
            unique_vals, indexes = torch.unique(ordinal_data, return_inverse=True)
            new_categories = torch.arange(int(feature['nclass']), dtype=torch.int64)
            mapped_categories = new_categories[indexes]
            
            thermometer = torch.zeros((data.shape[0], len(new_categories) + 1))
            thermometer[:, 0] = 1
            thermometer[torch.arange(data.shape[0]), 1 + mapped_categories] = -1
            thermometer = torch.cumsum(thermometer, dim=1)

            data_complete.append(thermometer[:, :-1])  # Exclude last column
            feat_names += [feature['name'] + "_" + str(j) for j in np.arange(len(new_categories))]

        elif feature['type'] == 'count':
            # Shift zero-based counts if necessary
            count_data = data[:, feat_idx].unsqueeze(1)
            if torch.min(count_data) == 0:
                count_data += 1
            data_complete.append(count_data)
            feat_names += [feature['name']]

        elif feature['type'].startswith('surv'):
            # Survival data take two columns
            data_complete.append(data[:, feat_idx : feat_idx + 2])
            feat_idx += 1
            feat_names += ["time", "censor"]
        
        else:
            # Keep continuous data as is
            data_complete.append(data[:, feat_idx].unsqueeze(1))
            feat_names += [feature['name']]
    
        feat_idx += 1
    # Concatenate processed features
    data = torch.cat(data_complete, dim=1)
    df = pd.DataFrame(data, columns=feat_names)

    # Read missing mask file
    n_samples, n_variables = data.shape[0], len(types_dict)
    miss_mask = torch.ones((n_samples, n_variables))

    if os.path.isfile(miss_file):
        with open(miss_file, 'r') as f:
            missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            missing_positions = torch.tensor(missing_positions, dtype=torch.long)
        if missing_positions.numel() != 0:
            miss_mask[missing_positions[:, 0] - 1, missing_positions[:, 1] - 1] = 0  # CSV indexes start at 1
    
    return df, types_dict, miss_mask, true_miss_mask, n_samples



def next_batch(data, types_dict, miss_mask, batch_size, index_batch):
    """
    Generates the next minibatch of data and splits it into its respective features.

    Parameters:
    -----------
    data : torch.Tensor
        The complete dataset from which to extract a batch.
    
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.
    
    miss_mask : torch.Tensor
        A binary mask indicating missing values (1 = observed, 0 = missing).
    
    batch_size : int
        The number of samples to include in each batch.
    
    index_batch : int
        The index of the current batch to extract.

    Returns:
    --------
    data_list : list of torch.Tensors
        A list containing feature-wise separated data for the current batch.
    
    miss_list : torch.Tensor
        The corresponding missing data mask for the current batch.
    """
    
    # Extract minibatch
    batch_xs = data[index_batch * batch_size : (index_batch + 1) * batch_size, :]
    
    # Split variables in the batch
    data_list, initial_index = [], 0
    for d in types_dict:
        dim = (int(d['nclass']) if d["type"] in ['cat', 'ordinal'] else int(d['dim']))
        data_list.append(batch_xs[:, initial_index : initial_index + dim])
        initial_index += dim
    
    # Extract missing mask for the batch
    miss_list = miss_mask[index_batch * batch_size : (index_batch + 1) * batch_size, :]
    
    return data_list, miss_list

def load_data_types(types_file):
    """
    Reads the types of data from a CSV file and returns a dictionary.

    Parameters:
    -----------
    types_file : str
        Path to the CSV file containing variable types.

    Returns:
    --------
    list of dict:
        A list where each dictionary specifies the type of a variable.
    """
    with open(types_file, newline='') as f:
        return [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]


def batch_normalization(batch_data_list, feat_types_list, miss_list):
    """
    Normalizes real-valued data while leaving categorical/ordinal variables unchanged.

    Parameters:
    -----------
    batch_data_list : list of torch.Tensor
        List of input data tensors, each corresponding to a feature.
    
    feat_types_list : list of dict
        List specifying the type of each feature.
    
    miss_list : torch.Tensor
        Binary mask indicating observed (1) and missing (0) values.

    Returns:
    --------
    normalized_data : list of torch.Tensor
        List of normalized feature tensors.
    
    normalization_parameters : list of tuples
        Normalization parameters for each feature.
    """

    normalized_data = []
    normalization_parameters = []

    for i, d in enumerate(batch_data_list):
        missing_mask = miss_list[:, i] == 0  # True for missing values, False for observed values
        observed_data = d[~missing_mask]  # Extract observed values

        feature_type = feat_types_list[i]['type']

        if feature_type == 'real':
            # Standard normalization (mean 0, std 1)
            if observed_data.numel() == 0:
                data_mean = torch.tensor(0.0, device=d.device, dtype=d.dtype)
                data_var = torch.tensor(1.0, device=d.device, dtype=d.dtype)
            else:
                data_var, data_mean = torch.var_mean(observed_data, unbiased=False)
                data_var = torch.clamp(data_var, min=1e-6, max=1e20)  # Prevent division by zero
            
            normalized_observed = (observed_data - data_mean) / torch.sqrt(data_var)
            normalized_d = torch.zeros_like(d)
            normalized_d[~missing_mask] = normalized_observed  # Assign transformed values
            normalized_d[missing_mask] = 0  # Missing values set to 0
            
            normalization_parameters.append((data_mean, data_var))

        elif feature_type == 'pos':
            # Log-normal transformation and normalization
            if observed_data.numel() == 0:
                observed_data_log = observed_data
                data_mean_log = torch.tensor(0.0, device=d.device, dtype=d.dtype)
                data_var_log = torch.tensor(1.0, device=d.device, dtype=d.dtype)
            else:
                observed_data_log = torch.log1p(observed_data)
                data_var_log, data_mean_log = torch.var_mean(observed_data_log, unbiased=False)
                data_var_log = torch.clamp(data_var_log, min=1e-6, max=1e20)

            normalized_observed = (observed_data_log - data_mean_log) / torch.sqrt(data_var_log)
            normalized_d = torch.zeros_like(d)
            normalized_d[~missing_mask] = normalized_observed
            normalized_d[missing_mask] = 0

            normalization_parameters.append((data_mean_log, data_var_log))

        elif feature_type == 'count':
            # Log transformation (No variance normalization)
            normalized_d = torch.zeros_like(d)
            normalized_d[~missing_mask] = torch.log1p(observed_data)  # Log-transform observed values
            normalized_d[missing_mask] = 0  # Missing values set to 0
            
            normalization_parameters.append((0.0, 1.0))

        elif feature_type == 'surv':
            # Log transformation (No variance normalization)
            if observed_data.numel() == 0:
                observed_data_log = observed_data[:, 0]
                data_mean_log = torch.tensor(0.0, device=d.device, dtype=d.dtype)
                data_var_log = torch.tensor(1.0, device=d.device, dtype=d.dtype)
            else:
                observed_data_log = torch.log1p(observed_data[:, 0])
                data_var_log, data_mean_log = torch.var_mean(observed_data_log, unbiased=False)
                data_var_log = torch.clamp(data_var_log, min=1e-6, max=1e20)

            normalized_observed = (observed_data_log - data_mean_log) / torch.sqrt(data_var_log)
            normalized_d = torch.zeros_like(d)
            normalized_d[~missing_mask][:, 0] = normalized_observed
            normalized_d[~missing_mask][:, 1] = observed_data[:, 1]
            normalized_d[missing_mask] = 0

            normalization_parameters.append((data_mean_log, data_var_log))

        elif feature_type.startswith('surv'):
            # Min max normalization
            if observed_data.numel() == 0:
                data_min = torch.tensor(0.0, device=d.device, dtype=d.dtype)
                data_max = torch.tensor(1.0, device=d.device, dtype=d.dtype)
            else:
                data_min = torch.min(observed_data[:, 0]) - 1e-3
                data_max = torch.max(observed_data[:, 0])
            normalization_parameters.append((data_min, data_max))
            
            
            denom = (data_max - data_min).clamp(min=1e-6)
            normalized_d = torch.zeros_like(d)
            normalized_d[~missing_mask, 0] = (observed_data[:, 0] - data_min) / denom
            normalized_d[~missing_mask, 1] = observed_data[:, 1]
            normalized_d[missing_mask] = 0  # Missing values set to 0


        else:
            # Keep categorical and ordinal values unchanged
            normalized_d = d.clone()
            normalization_parameters.append((0.0, 1.0))

        normalized_data.append(normalized_d)

    return normalized_data, normalization_parameters


def y_partition(samples_y, feat_types_list, y_dim_partition):
    """
    Partitions `samples_y` according to `y_dim_partition`.

    Parameters:
    -----------
    samples_y : torch.Tensor
        The latent variable `y` tensor of shape `(batch_size, sum(y_dim_partition))`.
    
    feat_types_list : list of dict
        List of dictionaries defining variable types and dimensions.
    
    y_dim_partition : list of int
        List specifying partition sizes for `y`.

    Returns:
    --------
    list of torch.Tensor :
        A list where each entry corresponds to a partitioned segment of `samples_y`.
    """
    
    partition_indices = np.insert(np.cumsum(y_dim_partition), 0, 0)
    
    return [samples_y[:, partition_indices[i]:partition_indices[i+1]] for i in range(len(feat_types_list))]


def discrete_variables_transformation(data, types_dict):
    """
    Transforms categorical and ordinal variables into their correct numerical representations.

    Parameters:
    -----------
    data : torch.Tensor
        The dataset containing mixed-type features.
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.

    Returns:
    --------
    torch.Tensor
        A tensor where categorical variables are mapped to their indices,
        and ordinal variables are transformed using sum-based encoding.
    """

    ind_ini, output = 0, []
    for d in types_dict:
        ind_end = ind_ini + (int(d['nclass']) if d["type"] in ['cat', 'ordinal'] else int(d['dim']))
        subset = data[:, ind_ini : ind_end]  # Extract relevant columns

        if d['type'] == 'cat':
            output.append(torch.argmax(subset, dim=1, keepdim=True))  # Argmax for categorical variables
        elif d['type'] == 'ordinal':
            output.append((torch.sum(subset, dim=1, keepdim=True) - 1))  # Sum-based transformation for ordinal variables
        else:
            output.append(subset)  # Keep continuous variables unchanged
        
        ind_ini = ind_end
    
    return torch.cat(output, dim=1)


def survival_variables_transformation(data, types_dict):
    """
    Transforms categorical and ordinal variables into their correct numerical representations.

    Parameters:
    -----------
    data : torch.Tensor
        The dataset containing mixed-type features.
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.

    Returns:
    --------
    torch.Tensor
        A tensor where categorical variables are mapped to their indices,
        and ordinal variables are transformed using sum-based encoding.
    """
    output = data.clone()

    feat_idx = 0
    for d in types_dict:
        if d['type'].startswith('surv'):
            subset = output[:, feat_idx : feat_idx + 2]
            time_cens = (torch.min(subset, dim=1, keepdim=True))
            output[:, feat_idx] = time_cens.values.squeeze(1)
            output[:, feat_idx + 1] = 1 - time_cens.indices.squeeze(1)
            feat_idx += 2
        else:
            feat_idx += 1
    
    return output


def encode_and_bind(df, feature):
    """
    One-hot encodes a categorical feature if it has more than 2 unique values.
    Drops the original column and appends the encoded dummies.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame.
        feature (str): The feature/column name to encode.
        
    Returns:
        pd.DataFrame: Modified DataFrame with encoding applied.
    """
    unique_values = df[feature].nunique()
    
    if unique_values > 2:
        dummies = pd.get_dummies(df[feature], drop_first=True, prefix=feature, prefix_sep='')
        df = pd.concat([df.drop(columns=[feature]), dummies], axis=1)
        
    return df



from torch.utils.data import Dataset

# class MyCustomDataset(Dataset):
#     def __init__(self, data_tensor, miss_mask_tensor):
#         self.data = data_tensor
#         self.miss = miss_mask_tensor

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, idx):
#         return self.data[idx], self.miss[idx]



class MyCustomDataset(Dataset):
    def __init__(self, data, miss_mask, types_dict):
        self.data = data
        self.miss_mask = miss_mask
        self.types_dict = types_dict
        # Precompute feature slice indices
        self.feature_slices = self._compute_feature_slices(types_dict)

    def _compute_feature_slices(self, types_dict):
        slices = []
        start = 0
        for d in types_dict:
            dim = int(d["nclass"]) if d["type"] in ['cat', 'ordinal'] else int(d["dim"])
            slices.append((start, start + dim))
            start += dim
        return slices

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]
        # miss_row = self.miss_mask[idx]

        # Split features
        data_list = [row[start:end] for start, end in self.feature_slices]
        # miss_list = [miss_row[start:end] for start, end in self.feature_slices]

        miss_list = self.miss_mask[idx, :]

        return data_list, miss_list


# ---------------------------------------------------------------------------
# V2A longitudinal data preparation
# ---------------------------------------------------------------------------

def prepare_longitudinal_tensors(longitudinal_df, patient_id_col='patient_id',
                                  time_col='visit_time', value_col='value',
                                  mask_col=None, n_patients=None):
    """
    Convert long-format longitudinal data to padded tensors aligned by patient row index.

    The patient IDs in *longitudinal_df[patient_id_col]* must be **0-based integer
    indices** that correspond to row positions in the baseline data table.

    Parameters
    ----------
    longitudinal_df : pd.DataFrame
        Long-format repeated-measures data.
    patient_id_col : str
        Column with 0-based patient row indices.
    time_col : str
        Column with visit times.
    value_col : str or list of str
        Column(s) with outcome values.  When a list is provided the returned
        *values_norm* tensor has shape ``(N, T_max, D)`` instead of ``(N, T_max)``.
    mask_col : str or None
        Optional column for per-visit observation mask. If None every visit is 1.
    n_patients : int or None
        Total number of patients (rows in baseline). Inferred from max id + 1 if None.

    Returns
    -------
    times_norm : torch.Tensor  (N, T_max)
        Visit times normalised to [0, 1].
    values_norm : torch.Tensor (N, T_max) or (N, T_max, D)
        Outcome values normalised to zero-mean unit-variance.
    masks : torch.Tensor       (N, T_max)
        Binary mask (1 = observed visit).
    norm_params : dict
        Keys: time_min, time_max, value_mean, value_std, max_visits.
        When value_col is a list, value_mean and value_std are lists (one per outcome).
    """
    # Determine whether we have single or multiple outcomes
    multi = isinstance(value_col, (list, tuple))
    value_cols = list(value_col) if multi else [value_col]
    n_outcomes = len(value_cols)

    patient_ids = longitudinal_df[patient_id_col].values.astype(int)
    N = n_patients if n_patients is not None else int(patient_ids.max()) + 1

    grouped = longitudinal_df.groupby(patient_id_col)
    max_visits = int(grouped.size().max())

    times = torch.zeros(N, max_visits)
    values = torch.zeros(N, max_visits, n_outcomes)
    masks = torch.zeros(N, max_visits)

    visit_count = np.zeros(N, dtype=int)
    for _, row in longitudinal_df.iterrows():
        pid = int(row[patient_id_col])
        v = visit_count[pid]
        if v < max_visits:
            times[pid, v] = float(row[time_col])
            for d, vc in enumerate(value_cols):
                values[pid, v, d] = float(row[vc])
            masks[pid, v] = float(row[mask_col]) if mask_col is not None else 1.0
            visit_count[pid] += 1

    # Normalise times to [0, 1]
    obs_mask = masks.bool()
    obs_times = times[obs_mask]
    time_min = obs_times.min().item() if obs_times.numel() > 0 else 0.0
    time_max = obs_times.max().item() if obs_times.numel() > 0 else 1.0
    time_range = max(time_max - time_min, 1e-6)
    times_norm = (times - time_min) / time_range * masks

    # Normalise values to zero-mean unit-variance (per outcome)
    value_means = []
    value_stds = []
    masks_exp = masks.unsqueeze(-1).expand_as(values)           # (N, T, D)
    for d in range(n_outcomes):
        obs_vals = values[:, :, d][obs_mask]
        vmean = obs_vals.mean().item() if obs_vals.numel() > 0 else 0.0
        vstd = max(obs_vals.std().item(), 1e-6) if obs_vals.numel() > 0 else 1.0
        values[:, :, d] = (values[:, :, d] - vmean) / vstd
        value_means.append(vmean)
        value_stds.append(vstd)

    values_norm = values * masks_exp

    # Backward compat: squeeze last dim when single outcome
    if not multi:
        values_norm = values_norm.squeeze(-1)                   # (N, T)
        value_means = value_means[0]
        value_stds = value_stds[0]

    norm_params = {
        'time_min': time_min, 'time_max': time_max,
        'value_mean': value_means, 'value_std': value_stds,
        'max_visits': max_visits,
    }
    return times_norm, values_norm, masks, norm_params
    

# ======================================================================
# Phase 1: Frozen global normalization (replaces per-batch normalization)
# ======================================================================

def compute_global_normalization(data_tensor, feat_types_list, miss_mask):
    """Compute normalization statistics once from the full training set.

    Parameters
    ----------
    data_tensor : torch.Tensor
        Full training data (N, D_encoded) after read_data encoding.
    feat_types_list : list of dict
        Feature type specifications.
    miss_mask : torch.Tensor
        Combined missing mask (N, n_features).

    Returns
    -------
    global_norm_params : list of tuples
        One (param1, param2) per feature, same format as batch_normalization.
    """
    # We need to split data into per-feature tensors the same way next_batch does
    feat_idx = 0
    global_norm_params = []
    for i, feature in enumerate(feat_types_list):
        ftype = feature['type']
        if ftype in ['cat', 'ordinal']:
            n_cols = int(feature.get('nclass', feature.get('dim', 1)))
        elif ftype.startswith('surv'):
            n_cols = 2
        else:
            n_cols = int(feature.get('dim', 1))

        feat_data = data_tensor[:, feat_idx:feat_idx + n_cols]
        observed_mask = miss_mask[:, i] == 1
        observed_data = feat_data[observed_mask]

        if ftype == 'real':
            if observed_data.numel() > 0:
                data_var, data_mean = torch.var_mean(observed_data, unbiased=False)
                data_var = torch.clamp(data_var, min=1e-6, max=1e20)
            else:
                data_mean, data_var = torch.tensor(0.0), torch.tensor(1.0)
            global_norm_params.append((data_mean, data_var))

        elif ftype == 'pos':
            if observed_data.numel() > 0:
                obs_log = torch.log1p(observed_data)
                data_var, data_mean = torch.var_mean(obs_log, unbiased=False)
                data_var = torch.clamp(data_var, min=1e-6, max=1e20)
            else:
                data_mean, data_var = torch.tensor(0.0), torch.tensor(1.0)
            global_norm_params.append((data_mean, data_var))

        elif ftype == 'count':
            global_norm_params.append((0.0, 1.0))

        elif ftype == 'surv':
            if observed_data.numel() > 0:
                obs_log = torch.log1p(observed_data[:, 0])
                data_var, data_mean = torch.var_mean(obs_log, unbiased=False)
                data_var = torch.clamp(data_var, min=1e-6, max=1e20)
            else:
                data_mean, data_var = torch.tensor(0.0), torch.tensor(1.0)
            global_norm_params.append((data_mean, data_var))

        elif ftype.startswith('surv'):
            if observed_data.numel() > 0:
                data_min = torch.min(observed_data[:, 0]) - 1e-3
                data_max = torch.max(observed_data[:, 0])
            else:
                data_min, data_max = torch.tensor(0.0), torch.tensor(1.0)
            global_norm_params.append((data_min, data_max))

        else:
            global_norm_params.append((0.0, 1.0))

        feat_idx += n_cols

    return global_norm_params


def batch_normalization_frozen(batch_data_list, feat_types_list, miss_list,
                                global_norm_params):
    """Apply normalization using frozen global statistics.

    Same interface as batch_normalization() but uses precomputed stats
    instead of computing per-batch statistics.

    Parameters
    ----------
    batch_data_list : list of torch.Tensor
    feat_types_list : list of dict
    miss_list : torch.Tensor
    global_norm_params : list of tuples from compute_global_normalization()

    Returns
    -------
    normalized_data : list of torch.Tensor
    normalization_parameters : list of tuples (same as global_norm_params)
    """
    normalized_data = []

    for i, d in enumerate(batch_data_list):
        missing_mask = miss_list[:, i] == 0
        observed_data = d[~missing_mask]
        feature_type = feat_types_list[i]['type']
        params = global_norm_params[i]

        if feature_type == 'real':
            data_mean, data_var = params
            if isinstance(data_mean, (int, float)):
                data_mean = torch.tensor(data_mean, device=d.device, dtype=d.dtype)
                data_var = torch.tensor(data_var, device=d.device, dtype=d.dtype)
            else:
                data_mean = data_mean.to(d.device)
                data_var = data_var.to(d.device)
            normalized_d = torch.zeros_like(d)
            if observed_data.numel() > 0:
                normalized_d[~missing_mask] = (observed_data - data_mean) / torch.sqrt(data_var)

        elif feature_type == 'pos':
            data_mean, data_var = params
            if isinstance(data_mean, (int, float)):
                data_mean = torch.tensor(data_mean, device=d.device, dtype=d.dtype)
                data_var = torch.tensor(data_var, device=d.device, dtype=d.dtype)
            else:
                data_mean = data_mean.to(d.device)
                data_var = data_var.to(d.device)
            normalized_d = torch.zeros_like(d)
            if observed_data.numel() > 0:
                obs_log = torch.log1p(observed_data)
                normalized_d[~missing_mask] = (obs_log - data_mean) / torch.sqrt(data_var)

        elif feature_type == 'count':
            normalized_d = torch.zeros_like(d)
            if observed_data.numel() > 0:
                normalized_d[~missing_mask] = torch.log1p(observed_data)

        elif feature_type == 'surv':
            data_mean, data_var = params
            if isinstance(data_mean, (int, float)):
                data_mean = torch.tensor(data_mean, device=d.device, dtype=d.dtype)
                data_var = torch.tensor(data_var, device=d.device, dtype=d.dtype)
            else:
                data_mean = data_mean.to(d.device)
                data_var = data_var.to(d.device)
            normalized_d = torch.zeros_like(d)
            if observed_data.numel() > 0:
                obs_log = torch.log1p(observed_data[:, 0])
                normalized_d[~missing_mask, 0] = (obs_log - data_mean) / torch.sqrt(data_var)
                normalized_d[~missing_mask, 1] = observed_data[:, 1]

        elif feature_type.startswith('surv'):
            data_min, data_max = params
            if isinstance(data_min, (int, float)):
                data_min = torch.tensor(data_min, device=d.device, dtype=d.dtype)
                data_max = torch.tensor(data_max, device=d.device, dtype=d.dtype)
            else:
                data_min = data_min.to(d.device)
                data_max = data_max.to(d.device)
            normalized_d = torch.zeros_like(d)
            if observed_data.numel() > 0:
                denom = (data_max - data_min).clamp(min=1e-6)
                normalized_d[~missing_mask, 0] = (observed_data[:, 0] - data_min) / denom
                normalized_d[~missing_mask, 1] = observed_data[:, 1]

        else:
            normalized_d = d.clone()

        normalized_data.append(normalized_d)

    return normalized_data, global_norm_params
