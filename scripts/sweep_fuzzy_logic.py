#!/usr/bin/env python
"""Grid sweep driver for the fuzzy-logic contrastive experiment.

Runs many :class:`ExperimentConfig` combinations of the existing runner
(:mod:`compgen.runners.fuzzy_logic_contrastive`) sequentially, one after the
other on a single device, and aggregates every result into one summary CSV.

The sweep is a cartesian product over ``GRID`` (below): each key is an
``ExperimentConfig`` field and each value is the list of settings to try. Edit
``GRID`` directly, or pass a JSON file of the same shape with ``--grid-json``.

Each combination gets its own output sub-directory named by a stable hash of
its overrides, so configurations that differ only in an optimization
hyperparameter (which the runner's ``run_name`` does not encode) never
overwrite each other. A combination is skipped on a re-run once all of its
per-variant result files already exist, which makes the sweep resumable; pass
``--force`` to re-run everything.

Examples::

    # list the combinations without running anything
    python scripts/sweep_fuzzy_logic.py --dry-run

    # run the default grid (cached results are skipped)
    python scripts/sweep_fuzzy_logic.py --sweep-name contrastive_v1

    # a smoke test of the first two combinations
    python scripts/sweep_fuzzy_logic.py --max-runs 2 --epochs 1

    # supply your own axes
    python scripts/sweep_fuzzy_logic.py --grid-json my_grid.json

Results land under ``results/fuzzy_logic_contrastive/sweeps/<sweep-name>/``:
one sub-directory per combination containing ``combo.json`` (the resolved
config and the overrides that produced it) plus one metrics JSON per variant,
and a ``summary.csv`` with one row per trained model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

# Make the repo root importable when the script is invoked directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compgen.runners.fuzzy_logic_contrastive.config import ExperimentConfig
from compgen.runners.fuzzy_logic_contrastive.run import run_experiment

# --------------------------------------------------------------------------- #
# The sweep grid: field name -> list of values. Cartesian product.
# --------------------------------------------------------------------------- #
# Default focuses on the contrastive design (attention architecture, lambda,
# representation) crossed with the optimization settings, matching the axes of
# interest. Add/remove axes freely; every key must be an ExperimentConfig field.
GRID: Dict[str, List[Any]] = {
    # Attention / contrastive design.
    "attention_types": [("softmax",), ("linear",)],
    "lambda_contrastive": [0.05, 0.2],
    "contrastive_representation": ["code", "row"],
    # Optimization.
    "learning_rate": [1e-3, 3e-4],
    "weight_decay": [0.1],
    "use_scheduler": [False, True],
}

# Preferred leading columns for the summary; remaining columns are appended
# alphabetically.
_LEADING_COLUMNS = [
    "combo",
    "attention_type",
    "lambda_contrastive",
    "contrastive_representation",
    "temperature",
    "learning_rate",
    "weight_decay",
    "use_scheduler",
    "task",
    "epochs",
    "seed",
    "test_mse",
    "test_r2",
    "test_within_005_accuracy",
    "ood_mse",
    "ood_r2",
    "id_mse",
    "train_loss",
    "test_contrastive_loss",
    "run_name",
]


# --------------------------------------------------------------------------- #
# Grid helpers
# --------------------------------------------------------------------------- #
def enumerate_grid(grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Return the cartesian product of ``grid`` as a list of override dicts."""
    keys = list(grid)
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*(grid[key] for key in keys))
    ]


def validate_grid(grid: Mapping[str, Sequence[Any]]) -> None:
    valid = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(grid) - valid)
    if unknown:
        raise SystemExit(
            f"Unknown grid axis/axes {unknown}. "
            f"Valid ExperimentConfig fields: {sorted(valid)}"
        )
    for key, values in grid.items():
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise SystemExit(f"Grid axis {key!r} must be a non-empty list of values")


def _canonical(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """JSON-serialisable, order-independent form of an override dict."""
    canonical: Dict[str, Any] = {}
    for key, value in overrides.items():
        if isinstance(value, tuple):
            canonical[key] = list(value)
        elif isinstance(value, list):
            canonical[key] = list(value)
        else:
            canonical[key] = value
    return canonical


def combo_id(overrides: Mapping[str, Any]) -> str:
    """Stable short hash identifying a combination of overrides."""
    payload = json.dumps(_canonical(overrides), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _describe(overrides: Mapping[str, Any]) -> str:
    return " ".join(f"{key}={_flatten(value)}" for key, value in overrides.items())


# --------------------------------------------------------------------------- #
# Config construction
# --------------------------------------------------------------------------- #
def build_base_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        task=args.task,
        num_train=args.num_train,
        num_eval=args.num_eval,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seq_len=args.seq_len,
        seed=args.seed,
        device=args.device,
        track=not args.no_track,
    )


def expected_variant_files(config: ExperimentConfig) -> List[str]:
    """Per-variant result filenames the runner will write for ``config``."""
    return [
        f"{variant.name}_{config.contrastive_representation}.json"
        for variant in config.variants()
    ]


# --------------------------------------------------------------------------- #
# Summary collection / writing
# --------------------------------------------------------------------------- #
def _flatten(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(_flatten(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def collect_rows(sweep_dir: Path) -> List[Dict[str, Any]]:
    """One row per variant, merging each combo's config with its metrics."""
    rows: List[Dict[str, Any]] = []
    for combo_dir in sorted(path for path in sweep_dir.iterdir() if path.is_dir()):
        combo_path = combo_dir / "combo.json"
        if not combo_path.exists():
            continue
        payload = json.loads(combo_path.read_text())
        config = {key: _flatten(value) for key, value in payload["config"].items()}
        for result_path in sorted(combo_dir.glob("*.json")):
            if result_path.name == "combo.json":
                continue
            metrics = json.loads(result_path.read_text())
            row: Dict[str, Any] = {"combo": combo_dir.name}
            row.update(config)
            row.update({key: _flatten(value) for key, value in metrics.items()})
            rows.append(row)
    return rows


def write_summary(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    present = {key for row in rows for key in row}
    columns = [col for col in _LEADING_COLUMNS if col in present]
    columns += sorted(present - set(columns))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def print_top(rows: Sequence[Mapping[str, Any]], count: int = 10) -> None:
    ranked = sorted(
        (row for row in rows if "test_mse" in row and row["test_mse"] != ""),
        key=lambda row: float(row["test_mse"]),
    )[:count]
    if not ranked:
        return
    print(f"\nTop {len(ranked)} by test_mse:")
    header = f"{'test_mse':>10}  {'test_r2':>8}  {'ood_mse':>10}  {'arch':<8} {'lambda':>7} {'repr':<6} {'lr':>8} {'wd':>5} {'sched':>5}"
    print(header)
    for row in ranked:
        print(
            f"{float(row['test_mse']):>10.4f}  "
            f"{float(row.get('test_r2', float('nan'))):>8.4f}  "
            f"{float(row.get('ood_mse', float('nan'))):>10.4f}  "
            f"{row.get('attention_type', ''):<8} "
            f"{row.get('lambda_contrastive', ''):>7} "
            f"{row.get('contrastive_representation', ''):<6} "
            f"{row.get('learning_rate', ''):>8} "
            f"{row.get('weight_decay', ''):>5} "
            f"{row.get('use_scheduler', ''):>5}"
        )


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description="Grid-sweep the fuzzy-logic contrastive experiment and "
        "aggregate results into a summary CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-name",
        default="sweep",
        help="sub-directory under --output-dir for this sweep's results",
    )
    parser.add_argument(
        "--output-dir",
        default="results/fuzzy_logic_contrastive/sweeps",
        help="base directory for sweep results",
    )
    parser.add_argument(
        "--grid-json",
        default=None,
        help="JSON file mapping ExperimentConfig field -> list of values, "
        "overriding the built-in GRID",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="only run the first N combinations (smoke tests)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the combinations and exit without running",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run combinations even if their results already exist",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="summary CSV path (default: <output-dir>/<sweep-name>/summary.csv)",
    )

    # Fixed (non-swept) experiment settings applied to every combination.
    parser.add_argument("--task", default=defaults.task)
    parser.add_argument("--num-train", type=int, default=defaults.num_train)
    parser.add_argument("--num-eval", type=int, default=defaults.num_eval)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--seq-len", type=int, default=defaults.seq_len)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="disable per-run logging through compgen.tracking",
    )
    return parser


def load_grid(args: argparse.Namespace) -> Dict[str, List[Any]]:
    if args.grid_json is None:
        return dict(GRID)
    payload = json.loads(Path(args.grid_json).read_text())
    if not isinstance(payload, dict):
        raise SystemExit("--grid-json must contain a JSON object of axis -> list")
    if not payload:
        raise SystemExit("--grid-json must define at least one axis")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grid = load_grid(args)
    validate_grid(grid)

    combos = enumerate_grid(grid)
    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    sweep_dir = Path(args.output_dir) / args.sweep_name
    print(
        f"Grid: {len(grid)} axes, {len(combos)} combinations "
        f"-> {sweep_dir}"
    )
    for index, overrides in enumerate(combos, start=1):
        print(f"  [{index:>3}/{len(combos)}] {_describe(overrides)}")
    if args.dry_run:
        print("Dry run: nothing executed.")
        return 0

    base = build_base_config(args)
    completed = skipped = failed = 0
    for index, overrides in enumerate(combos, start=1):
        config = replace(base, **overrides)
        combo_dir = sweep_dir / combo_id(overrides)
        config.output_dir = str(combo_dir)

        expected = expected_variant_files(config)
        if not args.force and expected and all(
            (combo_dir / name).exists() for name in expected
        ):
            print(f"[{index:>3}/{len(combos)}] skip (cached): {_describe(overrides)}")
            skipped += 1
            continue

        combo_dir.mkdir(parents=True, exist_ok=True)
        (combo_dir / "combo.json").write_text(
            json.dumps(
                {"config": asdict(config), "overrides": _canonical(overrides)},
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        print(f"[{index:>3}/{len(combos)}] run: {_describe(overrides)}")
        try:
            run_experiment(config)
            completed += 1
        except Exception as error:  # keep the sweep going past a bad combo
            failed += 1
            print(f"    FAILED: {type(error).__name__}: {error}", file=sys.stderr)

    rows = collect_rows(sweep_dir)
    summary_path = (
        Path(args.summary)
        if args.summary is not None
        else sweep_dir / "summary.csv"
    )
    write_summary(rows, summary_path)

    print(
        f"\nDone. ran={completed} skipped={skipped} failed={failed} "
        f"rows={len(rows)}"
    )
    if failed:
        print(f"{failed} combination(s) failed; see stderr above.", file=sys.stderr)
    print(f"Summary: {summary_path}")
    print_top(rows)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
