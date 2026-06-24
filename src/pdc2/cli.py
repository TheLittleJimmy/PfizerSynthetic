from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config, model_output_dir
from .data import load_pdc2_bundle, select_overfit_indices, subset_bundle
from .models import build_model
from .overfit import run_overfit_suite
from .training import evaluate_outputs, generate_longitudinal_samples, generate_prior_cohort, generate_static_samples, train_model


def _load_model_state_compat(model: torch.nn.Module, state: dict) -> None:
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    allowed_missing = ("u0_logsigma_head.",)
    bad_missing = [key for key in missing if not key.startswith(allowed_missing)]
    if bad_missing or unexpected:
        raise RuntimeError(
            "Checkpoint is not compatible with the current model. "
            f"bad_missing={bad_missing[:10]}, unexpected={unexpected[:10]}. "
            "For old deterministic-u0 checkpoints, load with strict=False so only "
            "the new u0_logsigma_head parameters are initialized from the current model."
        )


def _config_path(survival: str) -> Path:
    del survival
    return Path("configs") / "pdc2.yaml"


def train_command(args) -> None:
    cfg = load_config(args.config or _config_path(args.survival), {
        "dataset": {"name": args.dataset},
        "model": {"longitudinal_mode": "latent_ode", "survival": args.survival},
    })
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.output_dir is not None:
        cfg["dataset"]["output_root"] = str(Path(args.output_dir).resolve().parent)
    bundle = load_pdc2_bundle(cfg)
    if args.subset_size is not None:
        idx = select_overfit_indices(bundle, subset_size=args.subset_size, seed=cfg["training"]["seed"])
        bundle = subset_bundle(bundle, idx)
        cfg["training"]["subset_size"] = args.subset_size
    out = Path(args.output_dir) if args.output_dir else model_output_dir(cfg)
    result = train_model(bundle, cfg, output_dir=out)
    print(f"wrote {result['output_dir']}")


def evaluate_command(args) -> None:
    cfg = load_config(args.config or _config_path(args.survival), {
        "dataset": {"name": args.dataset},
        "model": {"longitudinal_mode": "latent_ode", "survival": args.survival},
    })
    if args.device is not None:
        cfg["training"]["device"] = args.device
    bundle = load_pdc2_bundle(cfg)
    model = build_model(bundle, cfg)
    checkpoint = Path(args.checkpoint) if args.checkpoint else model_output_dir(cfg) / "model_checkpoint.pt"
    state = torch.load(checkpoint, map_location=cfg["training"].get("device", "cpu"))
    _load_model_state_compat(model, state)
    device = torch.device(cfg["training"].get("device", "cpu"))
    model.to(device)
    out = model_output_dir(cfg)
    synthetic, latents = generate_static_samples(model, bundle, device, return_latents=True)
    synthetic_long = generate_longitudinal_samples(model, bundle, synthetic, device, latents=latents)
    metrics = evaluate_outputs(bundle, synthetic, synthetic_long, out / "figures")
    import json
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"evaluated {checkpoint}")


def generate_prior_command(args) -> None:
    cfg = load_config(args.config or _config_path(args.survival), {
        "dataset": {"name": args.dataset},
        "model": {"longitudinal_mode": "latent_ode", "survival": args.survival},
    })
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.n is not None:
        cfg["generation"]["prior_n"] = args.n
    if args.treatment is not None:
        cfg["generation"]["prior_treatment"] = args.treatment
    if args.time_grid is not None:
        cfg["generation"]["time_grid"] = [float(x) for x in args.time_grid.replace(",", " ").split() if x]
    if args.deterministic:
        cfg["generation"]["deterministic"] = True

    bundle = load_pdc2_bundle(cfg)
    model = build_model(bundle, cfg)
    checkpoint = Path(args.checkpoint) if args.checkpoint else model_output_dir(cfg) / "model_checkpoint.pt"
    state = torch.load(checkpoint, map_location=cfg["training"].get("device", "cpu"))
    _load_model_state_compat(model, state)
    device = torch.device(cfg["training"].get("device", "cpu"))
    model.to(device)
    output = Path(args.output_dir) if args.output_dir else model_output_dir(cfg) / "prior_generation"
    output.mkdir(parents=True, exist_ok=True)

    static_df, longitudinal_df, tensors = generate_prior_cohort(
        model,
        bundle,
        n=int(cfg["generation"]["prior_n"]),
        treatment=int(cfg["generation"]["prior_treatment"]),
        time_grid=cfg["generation"]["time_grid"],
        device=device,
        deterministic=bool(cfg["generation"].get("deterministic", False)),
        return_tensors=True,
    )
    static_df.to_csv(output / "prior_synthetic_static.csv", index=False)
    longitudinal_df.to_csv(output / "prior_synthetic_longitudinal.csv", index=False)
    metadata = {
        "mode": "prior",
        "n": int(cfg["generation"]["prior_n"]),
        "treatment": int(cfg["generation"]["prior_treatment"]),
        "time_grid": [float(x) for x in cfg["generation"]["time_grid"]],
        "baseline_generated_from_prior": bool(tensors["baseline_generated_from_prior"].item()),
        "uses_observed_future_outcomes": bool(tensors["uses_observed_future_outcomes"].item()),
        "checkpoint": str(checkpoint),
    }
    with open(output / "prior_generation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"wrote prior cohort to {output}")


def overfit_command(args) -> None:
    result = run_overfit_suite(
        dataset=args.dataset,
        subset_size=args.subset_size,
        seed=args.seed,
        survival=args.survival,
        config_path=args.config,
        settings=args.settings,
        max_visits=args.max_visits,
        device=args.device,
        epochs=args.epochs,
        write_summary=not args.skip_summary,
    )
    print(f"overfit {'passed' if result['passed'] else 'failed'}: {result['suite_dir']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDC2 longitudinal PhaseSyn runner")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["train", "evaluate"]:
        p = sub.add_parser(name)
        p.add_argument("--dataset", default="pdc2", choices=["pdc2", "pbc2"])
        p.add_argument("--survival", default="dynamic", choices=["dynamic"])
        p.add_argument("--config", default=None)
        if name == "train":
            p.add_argument("--epochs", type=int, default=None)
            p.add_argument("--device", default=None)
            p.add_argument("--subset-size", type=int, default=None)
            p.add_argument("--output-dir", default=None)
            p.set_defaults(func=train_command)
        else:
            p.add_argument("--checkpoint", default=None)
            p.add_argument("--device", default=None)
            p.set_defaults(func=evaluate_command)

    p = sub.add_parser("generate-prior")
    p.add_argument("--dataset", default="pdc2", choices=["pdc2", "pbc2"])
    p.add_argument("--survival", default="dynamic", choices=["dynamic"])
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--treatment", type=int, default=None)
    p.add_argument("--time-grid", default=None)
    p.add_argument("--deterministic", action="store_true")
    p.set_defaults(func=generate_prior_command)

    p = sub.add_parser("overfit")
    p.add_argument("--dataset", default="pdc2", choices=["pdc2", "pbc2"])
    p.add_argument("--survival", default="dynamic", choices=["dynamic"])
    p.add_argument("--subset-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config", default=None)
    p.add_argument("--settings", nargs="+", choices=["small", "medium", "large"], default=None)
    p.add_argument("--max-visits", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--skip-summary", action="store_true")
    p.set_defaults(func=overfit_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
