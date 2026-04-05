# Case study: Qwen3-32B inference on AMD MI300X VF

**Constraint:** single GPU VM, 1× AMD Instinct MI300X VF, ROCm 6.4.1
**Stack:** Kubernetes (RKE2) + AMD GPU Operator + AIRM + vLLM (ROCm backend)
**Model:** Qwen3-32B (fp16), served via OpenAI-compatible endpoint
**Full results:** [`results/mi300x-vf-qwen3-32b-rocm641.md`](../results/mi300x-vf-qwen3-32b-rocm641.md)

---

## Hypothesis

A single MI300X VF (191.69 GiB HBM3) has enough memory bandwidth and VRAM headroom
to serve a 32B-parameter model at useful throughput across moderate concurrency levels,
without OOM or significant latency degradation up to at least 4 concurrent requests.

---

## Measurement approach

1. Deploy Qwen3-32B via AIRM inference service (K8s pod with `amd.com/gpu: 1`)
2. Run OpenAI-compatible async load harness with concurrency sweep: 1, 2, 4, 8
3. Record: aggregate tok/s, per-request latency p50/p95/p99, TTFT (streaming)
4. Monitor VRAM live: `watch -n 1 'rocm-smi --showmeminfo vram'`

---

## Results summary

| Concurrency | tok/s | Latency p50 | Latency p95 | VRAM peak |
|-------------|-------|-------------|-------------|-----------|
| 1 | 54.0 | 2,371 ms | 2,377 ms | ~64 GiB |
| 2 | 99.0 | 2,584 ms | 2,594 ms | ~72 GiB |
| 4 | 181.6 | 2,810 ms | 2,869 ms | ~85 GiB |
| 8 | 295.8 | 3,093 ms | 3,129 ms | ~90 GiB |

**TTFT (streaming, p50):** 62 ms
**Error rate:** 0/112 requests across all tiers

---

## Key findings

**1. Throughput scales 3.4× from conc=1 to conc=4** — linear scaling to conc=4 shows the
HBM bandwidth (5.3 TB/s spec) is not the bottleneck at this model size and prompt length.
Compute bound for 32B fp16 weights at short context.

**2. Scaling efficiency drops at conc=8** — 5.48× aggregate throughput for 8× requests (68%
efficiency). Latency p50 rises from 2,810 ms to 3,093 ms — the KV-cache is approaching
the scheduler's batch capacity, not a memory OOM. No error or eviction events observed.

**3. 191.69 GiB HBM3 gives large headroom** — Qwen3-32B fp16 weights consume ~64 GiB.
At conc=8 with 128-token outputs, KV-cache peaks at ~90 GiB, leaving ~100 GiB free.
A second 32B model or longer context window (up to ~512K tokens) would fit.

**4. p99 TTFT spike (130 ms) is cold-start, not steady-state** — isolated to the first request
after an idle period. Subsequent p50 TTFT stable at 62 ms.

---

## What failed and what it taught us

**Issue: `amd-smi metric -m TEMPERATURE_JUNCTION` returned invalid-parameter**

On ROCm 6.4.1 this metric name is not accepted by `amd-smi`. Used `rocm-smi --showtemp`
instead. Takeaway: `amd-smi` and `rocm-smi` are not drop-in replacements on every ROCm
version — always verify flag names against the installed version before scripting.

**Issue: K8s GPU request/limit mismatch**

An early manifest set `requests: amd.com/gpu: 1` but omitted it from `limits`. The pod
scheduled but the GPU device was not visible inside the container. AMD GPU Operator (v1.4.1)
requires `amd.com/gpu` in both `requests` and `limits` — unlike some NVIDIA configurations
where only `limits` is sufficient.

```yaml
# Correct manifest pattern
resources:
  requests:
    amd.com/gpu: "1"
  limits:
    amd.com/gpu: "1"
```

**Issue: vLLM first-request latency 3× baseline**

Normal vLLM behaviour — weights are paged into HBM on first prefill. Resolved by sending
one warm-up request before starting the timed benchmark. All results above exclude warm-up.

---

## SLO assessment

| SLO | Target | Measured | Pass? |
|-----|--------|----------|-------|
| TTFT p95 (streaming) | < 200 ms | 84 ms | ✓ |
| Latency p95 (non-streaming, short) | < 1,000 ms | 621 ms | ✓ |
| Latency p95 (non-streaming, long) | < 5,000 ms | 2,669 ms | ✓ |
| Aggregate throughput at conc=4 | > 100 tok/s | 181.6 tok/s | ✓ |
| Error rate | 0% | 0% | ✓ |
| OOM events | 0 | 0 | ✓ |

All SLOs met at conc ≤ 4 on a single MI300X VF with Qwen3-32B fp16.

---

## Limitations

- **Single VF:** results are for one virtualised GPU slice; bare-metal MI300X would show
  higher FP32/FP16 FLOPS and potentially tighter latency at the same model size.
- **Single model:** only Qwen3-32B tested; smaller models (8B, 14B) would scale to higher
  concurrency before reaching the batch-capacity inflection point.
- **Fixed prompt length:** concurrency sweep used 128-token max outputs; longer context
  (4K+ tokens) would shift the VRAM and latency profile significantly.
- **No tensor parallelism:** single GPU — TP=1 only. Multi-GPU TP would reduce per-GPU
  memory and improve throughput for very large models.
