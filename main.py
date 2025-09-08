#!/usr/bin/env python3
"""
Unified CLI entry point for running explanations.

Examples:
  python main.py --model schnet --method gnnexplainer
  python main.py --model schnet --method 3dgraphx_t
  python main.py --model dimenet --method 3dgraphx_t
  python main.py --model schnet --method pgexplainer
  python main.py --model schnet --method lri_bern
  python main.py --model dimenet --method gnnexplainer

Notes:
- This CLI dispatches to existing runner modules under `three_d_graphx`.
- Many underlying scripts manage their own hyperparameters; this CLI
  currently isolates their argument parsing by resetting sys.argv to
  avoid conflicts. Additional flags here are for future integration.
"""

import argparse
from typing import Dict, Tuple
import torch
import os.path as osp
from torch_geometric.datasets import QM9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3D graph explainers")

    # Core selection
    parser.add_argument(
        "--model",
        choices=["schnet", "dimenet"],
        default="schnet",
        help="Backbone model",
    )
    parser.add_argument(
        "--method",
        choices=[
            "gnnexplainer",
            "pgexplainer",
            "lri_bern",
            "3dgraphx_t",
            "3dgraphx_i",
        ],
        default="3dgraphx_t",
        help="Explanation method",
    )

    # Dataset/config (currently runners are QM9-specific)
    parser.add_argument(
        "--dataset",
        choices=["qm9"],
        default="qm9",
        help="Dataset to use",
    )
    parser.add_argument(
        "--property",
        type=str,
        default=0,
        required=False,
        help="QM9 property name or index (runner defaults if unset)",
    )

    # Compute & training knobs (not all wired into underlying scripts yet)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cuda", help="Compute device"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit test samples")
    parser.add_argument(
        "--dimenet-plus-plus",
        action="store_true",
        help="Use DimeNet++ where supported",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path for parametric explainers",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model
    method = args.method

    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "QM9")
    dataset = QM9(path)
    if args.property is not None:
        if args.property in dataset.raw_file_names:
            args.property = dataset.raw_file_names.index(args.property)
        else:
            try:
                args.property = int(args.property)
            except ValueError:
                raise ValueError(f"Invalid QM9 property: {args.property}")
    else:
        args.property = 0

    device = args.device

    import os

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
    else:
        raise ValueError(f"Invalid device: {device}")

    exp_budge = list(range(2, 10))
    exp_loss = {k: [] for k in exp_budge}

    module = MODULE_MAP[key]

    # Prepare module-specific argv if needed (keep minimal to avoid conflicts)
    module_argv: list[str] = []
    if args.model == "dimenet" and args.method == "3dgraphx_t":
        # graph_t_dimnet.py supports --use_dimenet_plus_plus
        if args.dimenet_plus_plus:
            module_argv.append("--use_dimenet_plus_plus")

    print(
        f"Running module '{module}' (dataset={args.dataset}, device={args.device or 'auto'})"
    )

    dispatch(module, module_argv)


if __name__ == "__main__":
    main()
