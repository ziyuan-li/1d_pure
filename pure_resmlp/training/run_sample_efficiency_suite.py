"""Distribute the sample-efficiency grid across GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from pure_resmlp.paths import ARTIFACTS_DIR, RUNS_DIR


FRACTIONS = {
    0.10: "jocs_sample_efficiency_f010_v1",
    0.25: "jocs_sample_efficiency_f025_v1",
    0.50: "jocs_sample_efficiency_f050_v1",
}
DATASET_MODELS = {
    "friedman": ("mlp", "resmlp", "rv_pure", "cv_pure"),
    "concrete": ("mlp", "resmlp", "rv_pure", "cv_pure"),
    "parkinsons_telemonitoring": (
        "mlp_ordinary",
        "mlp_context",
        "mlp_explicit_cross",
        "dcnv2_context",
        "rv_pure",
        "cv_pure",
    ),
}


@dataclass(frozen=True)
class Job:
    fraction: float
    experiment: str
    dataset: str
    model: str
    seed: int


def jobs(seeds: tuple[int, ...]) -> list[Job]:
    return [
        Job(fraction, experiment, dataset, model, seed)
        for fraction, experiment in FRACTIONS.items()
        for dataset, models in DATASET_MODELS.items()
        for model in models
        for seed in seeds
    ]


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
    log_path = log_dir / f"gpu_{gpu}.log"
    with log_path.open("a", encoding="utf-8") as log:
        for job in queue:
            run_dir = (
                runs_root
                / job.experiment
                / job.dataset
                / job.model
                / f"seed_{job.seed}"
            )
            if (run_dir / "manifest.json").is_file():
                completed += 1
                continue
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
                "--train-fraction",
                str(job.fraction),
                "--subset-seed",
                "0",
                "--device",
                "cuda:0",
                "--output-root",
                str(runs_root),
            ]
            if run_dir.exists():
                command.append("--overwrite")
            if job.dataset == "parkinsons_telemonitoring":
                command.extend(
                    [
                        "--epochs",
                        "100",
                        "--patience",
                        "15",
                        "--batch-size",
                        "256",
                        "--hidden-dim",
                        "64",
                    ]
                )
            else:
                command.extend(
                    ["--epochs", "200", "--patience", "20", "--batch-size", "128"]
                )
            log.write(
                f"START fraction={job.fraction} dataset={job.dataset} "
                f"model={job.model} seed={job.seed}\n"
            )
            log.flush()
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode == 0:
                completed += 1
            else:
                failed += 1
                log.write(f"FAILED returncode={result.returncode}\n")
                log.flush()
            print(
                f"gpu={gpu} finished={completed + failed}/{len(queue)} "
                f"failed={failed}",
                flush=True,
            )
    return completed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--training-seeds", type=int, nargs="+", default=tuple(range(5)))
    parser.add_argument("--runs-root", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ARTIFACTS_DIR / "logs" / "sample_efficiency",
    )
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU identifiers must be unique")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    all_jobs = jobs(tuple(args.training_seeds))
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
    print(f"sample-efficiency suite complete: completed={completed} failed={failed}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
