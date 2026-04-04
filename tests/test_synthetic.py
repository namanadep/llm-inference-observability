import json

from infer_harness.synthetic import run_synthetic_sweep


def test_synthetic_writes_jsonl(tmp_path):
    cfg = {"tensor_parallel": [1], "batch_size": [2], "precision": ["bf16"], "warmup_iterations": 1, "timed_iterations": 2}
    out = tmp_path / "r.jsonl"
    run_synthetic_sweep(cfg, out)
    line = out.read_text().strip().splitlines()[0]
    d = json.loads(line)
    assert d["event"] == "synthetic"
    assert "tokens_per_s" in d
