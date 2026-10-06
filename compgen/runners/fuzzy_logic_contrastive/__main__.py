"""Command-line entry point for the fuzzy-logic four-model contrastive experiment.

Run with::

    python -m compgen.runners.fuzzy_logic_contrastive

All fields of :class:`~compgen.runners.fuzzy_logic_contrastive.config.ExperimentConfig`
that shape a run are exposed as flags; the small defaults are used when omitted.
"""

import argparse
from typing import List, Optional

from compgen.runners.fuzzy_logic_contrastive.config import (
    CONTRASTIVE_REPRESENTATIONS,
    ExperimentConfig,
)
from compgen.runners.fuzzy_logic_contrastive.run import run_experiment


def build_parser() -> argparse.ArgumentParser:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description="Compare softmax and linear attention, with and without the "
        "response-token attention-code contrastive term, on the fuzzy-logic task.",
    )
    parser.add_argument("--task", default=defaults.task)
    parser.add_argument("--num-train", type=int, default=defaults.num_train)
    parser.add_argument("--num-eval", type=int, default=defaults.num_eval)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument(
        "--lambda-contrastive", type=float, default=defaults.lambda_contrastive
    )
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument(
        "--contrastive-representation",
        choices=CONTRASTIVE_REPRESENTATIONS,
        default=defaults.contrastive_representation,
        help="final-layer attention reduction for the contrastive term "
        "(default: code)",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument(
        "--attention-types",
        nargs="+",
        default=list(defaults.attention_types),
        help="attention architectures to compare (default: softmax linear)",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="disable logging runs through compgen.tracking",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        task=args.task,
        num_train=args.num_train,
        num_eval=args.num_eval,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lambda_contrastive=args.lambda_contrastive,
        temperature=args.temperature,
        contrastive_representation=args.contrastive_representation,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        attention_types=tuple(args.attention_types),
        track=not args.no_track,
    )


def main(argv: Optional[List[str]] = None) -> List[dict]:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    results = run_experiment(config)
    for result in results:
        print(
            f"{result['run_name']}: "
            f"test_mse={result['test_mse']:.4f} "
            f"test_r2={result['test_r2']:.4f} "
            f"ood_mse={result['ood_mse']:.4f} "
            f"contrastive={result['test_contrastive_loss']:.4f}"
        )
    print(f"Wrote {len(results)} result records under {config.output_dir}")
    return results


if __name__ == "__main__":
    main()