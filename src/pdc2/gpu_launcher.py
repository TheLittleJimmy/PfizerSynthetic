from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import load_config


ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ["small", "medium", "large"]
SURVIVALS = ["dynamic"]


@dataclass
class Job:
    name: str
    cmd: list[str]
    gpu: int | None = None
    log_path: Path | None = None
    returncode: int | None = None


def available_gpus() -> list[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def run_job_batch(jobs: list[Job], gpus: list[int]) -> list[Job]:
    if not jobs:
        return []
    logs_dir = ROOT / "outputs" / "pdc2" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    active: list[tuple[Job, subprocess.Popen, object]] = []
    pending = list(jobs)
    free_gpus = list(gpus) if gpus else [None]
    completed: list[Job] = []

    while pending or active:
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            job = pending.pop(0)
            job.gpu = gpu
            job.log_path = logs_dir / f"{job.name}.log"
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "2")
            env.setdefault("MKL_NUM_THREADS", "2")
            python_paths = [str(ROOT / "src"), str(ROOT)]
            if env.get("PYTHONPATH"):
                python_paths.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(python_paths)
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_handle = open(job.log_path, "w", encoding="utf-8")
            log_handle.write(f"$ {' '.join(job.cmd)}\n")
            log_handle.write(f"GPU={gpu}\n\n")
            log_handle.flush()
            proc = subprocess.Popen(job.cmd, cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            active.append((job, proc, log_handle))
            print(f"started {job.name} on GPU {gpu}; log={job.log_path}", flush=True)
        time.sleep(2)
        still_active = []
        for job, proc, log_handle in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((job, proc, log_handle))
                continue
            job.returncode = rc
            log_handle.write(f"\nreturncode={rc}\n")
            log_handle.close()
            completed.append(job)
            free_gpus.append(job.gpu)
            print(f"finished {job.name} rc={rc}", flush=True)
        active = still_active
    return completed


def model_dir(survival: str) -> Path:
    del survival
    return ROOT / "outputs" / "pdc2" / "model"


def collect_overfit_summary(survival: str, jobs: list[Job], subset_size: int, seed: int) -> tuple[pd.DataFrame, bool]:
    rows = []
    job_rc = {job.name.split("_")[-1]: job.returncode for job in jobs}
    for setting in SETTINGS:
        out_dir = model_dir(survival) / "overfit" / setting
        metrics_path = out_dir / "metrics.json"
        diag_path = out_dir / "overfit_diagnostics.json"
        passed = False
        loss_drop = 0.0
        rmse_ratio = 0.0
        event_diff = 0.0
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            rmse_ratio = float(metrics.get("continuous_rmse_ratio", 0.0))
            event_diff = float(metrics.get("event_rate_diff", 0.0))
        if diag_path.exists():
            with open(diag_path, "r", encoding="utf-8") as f:
                diag = json.load(f)
            gate = diag.get("gate", {})
            passed = bool(gate.get("passed", False))
            loss_drop = float(gate.get("loss_decrease_ratio", 0.0))
        rows.append({
            "setting": setting,
            "passed": passed,
            "returncode": job_rc.get(setting),
            "loss_decrease_ratio": loss_drop,
            "continuous_rmse_ratio": rmse_ratio,
            "event_rate_diff": event_diff,
            "output_dir": str(out_dir),
        })
    summary = pd.DataFrame(rows)
    suite_dir = model_dir(survival) / "overfit"
    suite_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(suite_dir / "summary.csv", index=False)
    passed_all = bool(summary["passed"].all())
    with open(suite_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# PDC2 PhaseSyn Overfit Summary\n\n")
        f.write(f"Survival: `{survival}`  \n")
        f.write(f"Subset size: `{subset_size}` seed `{seed}`  \n\n")
        try:
            f.write(summary.to_markdown(index=False))
        except Exception:
            f.write(summary.to_string(index=False))
        f.write("\n\n")
        f.write("Gate: **passed**\n" if passed_all else "Gate: **failed**\n")
    return summary, passed_all


def overfit_jobs(survival: str, subset_size: int, seed: int, max_visits: int | None) -> list[Job]:
    jobs = []
    for setting in SETTINGS:
        cmd = [
            sys.executable,
            "-m", "pdc2.cli",
            "overfit",
            "--dataset", "pdc2",
            "--survival", survival,
            "--subset-size", str(subset_size),
            "--seed", str(seed),
            "--settings", setting,
            "--device", "cuda",
            "--skip-summary",
        ]
        if max_visits is not None:
            cmd.extend(["--max-visits", str(max_visits)])
        jobs.append(Job(name=f"model_{survival}_{setting}", cmd=cmd))
    return jobs


def train_job(survival: str, max_visits: int | None) -> Job:
    cmd = [
        sys.executable,
        "-m", "pdc2.cli",
        "train",
        "--dataset", "pdc2",
        "--survival", survival,
        "--device", "cuda",
    ]
    if max_visits is not None:
        cfg = load_config("configs/pdc2.yaml")
        cfg["dataset"]["max_visits"] = int(max_visits)
        temp = ROOT / "outputs" / "pdc2" / "logs" / f"model_{survival}_train_config.yaml"
        temp.parent.mkdir(parents=True, exist_ok=True)
        import yaml
        with open(temp, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        cmd.extend(["--config", str(temp)])
    return Job(name=f"model_{survival}_train", cmd=cmd)


def write_report(lines: list[str]) -> Path:
    report = ROOT / "outputs" / "pdc2" / "reports" / "migration_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_all_gpu(subset_size: int = 32, seed: int = 1, max_visits: int | None = None) -> Path:
    gpus = available_gpus()
    if not gpus:
        raise RuntimeError("No GPUs found by nvidia-smi.")
    report_lines = [
        "# PDC2 PhaseSyn GPU Experiment Report",
        "",
        f"Started: {datetime.now().isoformat(timespec='seconds')}",
        f"GPUs: `{', '.join(map(str, gpus))}`",
        "",
        "## PhaseSyn Model",
    ]
    write_report(report_lines)

    jobs = []
    for survival in SURVIVALS:
        jobs.extend(overfit_jobs(survival, subset_size, seed, max_visits))
    completed = run_job_batch(jobs, gpus)

    gates: dict[str, bool] = {}
    for survival in SURVIVALS:
        survival_jobs = [job for job in completed if job.name.startswith(f"model_{survival}_")]
        _, passed = collect_overfit_summary(survival, survival_jobs, subset_size, seed)
        gates[survival] = passed
        report_lines.append(f"{survival} overfit gate: {'passed' if passed else 'failed'}")
        report_lines.append(f"{survival} overfit summary: `{model_dir(survival) / 'overfit' / 'summary.md'}`")

    if not gates.get("dynamic", False):
        report_lines.append("Canonical dynamic survival overfit failed; full PhaseSyn training skipped.")
        return write_report(report_lines)

    train_jobs = [train_job("dynamic", max_visits)]
    completed_train = run_job_batch(train_jobs, gpus)
    for job in completed_train:
        survival = "dynamic"
        report_lines.append(f"{survival} full train rc={job.returncode}; output `{model_dir(survival)}`")
    report_lines.append("")
    report_lines.append(f"Finished: {datetime.now().isoformat(timespec='seconds')}")
    return write_report(report_lines)


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run PDC2 PhaseSyn experiments on available GPUs")
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-visits", type=int, default=None)
    args = parser.parse_args(argv)
    report = run_all_gpu(args.subset_size, args.seed, args.max_visits)
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
