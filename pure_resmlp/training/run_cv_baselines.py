"""Run the CV-MLP and CV-PURe comparison grid."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from pure_resmlp.paths import ARTIFACTS_DIR, RUNS_DIR


CV_BASELINE_EXPERIMENT = "jocs_complex_architecture_control_v1"
BIKE_EXPERIMENT = "jocs_bike_context_v1"


@dataclass(frozen=True)
class Job:
    dataset: str
    seed: int
    model: str
    experiment: str


def jobs() -> list[Job]:
    return [
        *(
            Job("synthetic_higher_order", seed, "cv_mlp", CV_BASELINE_EXPERIMENT)
            for seed in range(3)
        ),
        *(
            Job("parkinsons_telemonitoring", seed, "cv_mlp", CV_BASELINE_EXPERIMENT)
            for seed in range(5)
        ),
        *(
            Job("bike_sharing", seed, "cv_mlp", CV_BASELINE_EXPERIMENT)
            for seed in range(5)
        ),
        *(
            Job("bike_sharing", seed, "cv_pure", BIKE_EXPERIMENT)
            for seed in range(5)
        ),
    ]


def _run(command: list[str], *, environment: dict[str, str], log) -> bool:
    return (
        subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
        == 0
    )


def _matches_frozen_protocol(run_dir: Path, job: Job) -> bool:
    """Check that a saved run matches the configured protocol."""
    config_path = run_dir / "config.json"
    manifest_path = run_dir / "manifest.json"
    if not config_path.is_file() or not manifest_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    parameters = config.get("parameters", {})
    return (
        config.get("experiment") == job.experiment
        and config.get("dataset") == job.dataset
        and config.get("model") == job.model
        and config.get("seed") == job.seed
        and config.get("split_seed") == 0
        and parameters.get("hidden_dim") == 64
        and parameters.get("epochs") == 100
        and parameters.get("patience") == 15
        and parameters.get("batch_size") == 256
        and parameters.get("output_head") == "learnable_real_projection_v1"
    )


def _ensure_interactions(
    run_dir: Path,
    dataset: str,
    *,
    environment: dict[str, str],
    log,
) -> bool:
    base = [
        sys.executable,
        "-m",
        "pure_resmlp.evaluation.run_interactions",
        str(run_dir),
        "--n-samples",
        "256",
        "--device",
        "cuda:0",
    ]
    if not (run_dir / "interaction_metrics.json").is_file():
        if not _run(base + ["--seed", "0"], environment=environment, log=log):
            return False
    if dataset == "synthetic_higher_order" and not (
        run_dir / "interaction_robustness.json"
    ).is_file():
        if not _run(
            base + ["--resample-seeds", "0", "1", "2", "3", "4"],
            environment=environment,
            log=log,
        ):
            return False
    return True


def _run_queue(
    gpu: int,
    queue: list[Job],
    *,
    runs_root: Path,
    log_dir: Path,
) -> tuple[int, int]:
    completed = 0
    failed = 0
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with (log_dir / f"gpu_{gpu}.log").open("a", encoding="utf-8") as log:
        for job in queue:
            run_dir = (
                runs_root
                / job.experiment
                / job.dataset
                / job.model
                / f"seed_{job.seed}"
            )
            command = [
                sys.executable,
                "-m",
                "pure_resmlp.training.run_experiment",
                "--experiment",
                job.experiment,
                "--dataset",
                job.dataset,
                "--model",
                job.model,
                "--seed",
                str(job.seed),
                "--split-seed",
                "0",
                "--hidden-dim",
                "64",
                "--epochs",
                "100",
                "--patience",
                "15",
                "--batch-size",
                "256",
                "--device",
                "cuda:0",
                "--output-root",
                str(runs_root),
            ]
            success = True
            if not _matches_frozen_protocol(run_dir, job):
                if run_dir.exists():
                    command.append("--overwrite")
                success = _run(command, environment=environment, log=log)
            if success:
                success = _ensure_interactions(
                    run_dir, job.dataset, environment=environment, log=log
                )
            if success:
                completed += 1
            else:
                failed += 1
            print(
                f"gpu={gpu} finished={completed + failed}/{len(queue)} failed={failed}",
                flush=True,
            )
    return completed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--runs-root", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ARTIFACTS_DIR / "logs" / "cv_baselines",
    )
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU identifiers must be unique")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    all_jobs = jobs()
    queues = [all_jobs[index :: len(args.gpus)] for index in range(len(args.gpus))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(
                _run_queue,
                gpu,
                queue,
                runs_root=args.runs_root,
                log_dir=args.log_dir,
            )
            for gpu, queue in zip(args.gpus, queues, strict=True)
        ]
        results = [future.result() for future in futures]
    completed = sum(result[0] for result in results)
    failed = sum(result[1] for result in results)
    print(f"CV baseline suite complete: completed={completed} failed={failed}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
