"""Optional PyTorch + transformers path (install with .[torch])."""

from __future__ import annotations


def run_torch_stub(model_name: str, out: str) -> None:
    """Placeholder: extend with real generate() + timing."""
    raise NotImplementedError(
        f"Install infer-harness[torch] and implement generate loop for {model_name}; write JSONL to {out}"
    )
