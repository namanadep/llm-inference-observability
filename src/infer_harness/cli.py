"""infer-harness CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    p = argparse.ArgumentParser(prog="infer-harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synthetic", help="Deterministic synthetic sweep (no GPU)")
    s.add_argument("--config", type=Path, default=Path("configs/sweep.yaml"))
    s.add_argument("--out", type=Path, default=Path("results/run.jsonl"))

    t = sub.add_parser("torch", help="PyTorch path (implement in torch_runner.py)")
    t.add_argument("--model", required=True)
    t.add_argument("--out", type=Path, default=Path("results/torch.jsonl"))

    args = p.parse_args()
    if args.cmd == "synthetic":
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}
        from infer_harness.synthetic import run_synthetic_sweep

        run_synthetic_sweep(cfg, args.out)
        print(f"Wrote {args.out}")
    elif args.cmd == "torch":
        from infer_harness.torch_runner import run_torch_stub

        run_torch_stub(args.model, str(args.out))


if __name__ == "__main__":
    main()
