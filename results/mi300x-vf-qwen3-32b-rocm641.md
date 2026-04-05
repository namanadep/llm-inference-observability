# Inference results — Qwen3-32B on AMD Instinct MI300X VF

**Date:** 2026-03-29
**Model:** `Qwen/Qwen3-32B`
**Serving stack:** AMD AI Resource Manager (AIRM / AIM) → vLLM backend → OpenAI-compatible endpoint
**GPU:** AMD Instinct MI300X VF (gfx942 / CDNA3, 191.69 GiB HBM3)
**ROCm:** 6.4.1
**PyTorch:** 2.4.1+rocm6.0
**Kubernetes:** RKE2 v1.34.1+rke2r1, AMD GPU Operator v1.4.1, resource `amd.com/gpu: 1`
**Benchmark script:** `src/infer_harness.py` (OpenAI-compatible client, async concurrency sweep)
**Endpoint:** `https://airmui.<domain>.nip.io/v1/chat/completions`

---

## Stack diagram

```
┌─────────────────────────────────────────────────┐
│  infer_harness.py  (async OpenAI client)         │
│  concurrency sweep: 1 → 2 → 4 → 8               │
└────────────────┬────────────────────────────────┘
                 │  HTTPS / OpenAI API
┌────────────────▼────────────────────────────────┐
│  AMD AI Resource Manager (AIRM)                  │
│  Kubernetes Service  amd.com/gpu: 1              │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  vLLM  (ROCm backend)                            │
│  Model: Qwen3-32B  (fp16 weights)                │
│  GPU:   MI300X VF  191.69 GiB HBM3              │
└─────────────────────────────────────────────────┘
```

---

## Latency — non-streaming (single request baseline)

### Short prompt (39 chars, max_tokens=32)

| Metric | Value |
|--------|-------|
| n | 20 requests |
| Mean latency | **617.7 ms** |
| p50 | 617.4 ms |
| p95 | 620.8 ms |
| p99 | 622.5 ms |
| Min / Max | 614.0 / 623.0 ms |
| Tokens / request | 32 |

### Long prompt (4,000 chars, max_tokens=128)

| Metric | Value |
|--------|-------|
| n | 10 requests |
| Mean latency | **2,639 ms** |
| p50 | 2,633 ms |
| p95 | 2,669 ms |
| p99 | 2,671 ms |
| Min / Max | 2,607 / 2,671 ms |

**Observation:** p95–p99 spread is <5 ms in both cases — stable serving with no GPU memory pressure events during this run.

---

## TTFT — streaming (short prompt, max_tokens=64)

Time to first token (SSE stream, first non-empty delta):

| Metric | Value |
|--------|-------|
| n | 16 requests |
| Mean TTFT | **67.2 ms** |
| p50 | 62.0 ms |
| p95 | 84.0 ms |
| p99 | 130.6 ms |
| Min / Max | 61.2 / 142.2 ms |

**p99 spike to 130 ms:** isolated — one request; likely KV-cache eviction on first prefill of a cold model state. p50/p95 are stable.

---

## Throughput sweep — concurrency 1 → 8

| Concurrency | Requests | tok/s (aggregate) | Req/s | Latency p50 (ms) | Latency p95 (ms) | Errors |
|-------------|----------|-------------------|-------|------------------|------------------|--------|
| **1** | 28 | **54.0** | 0.42 | 2,371 | 2,377 | 0 |
| **2** | 28 | **99.0** | 0.77 | 2,584 | 2,594 | 0 |
| **4** | 28 | **181.6** | 1.38 | 2,810 | 2,869 | 0 |
| **8** | 28 | **295.8** | 2.06 | 3,093 | 3,129 | 0 |

**Model:** Qwen3-32B, max_tokens=128, same prompt batch across all concurrency levels.

### Throughput scaling

```
conc=1  ████████░░░░░░░░░░░░░░░░░░░░░░  54 tok/s
conc=2  ███████████████░░░░░░░░░░░░░░░  99 tok/s  (1.83×)
conc=4  ████████████████████████████░░  182 tok/s (3.36×)
conc=8  ██████████████████████████████  296 tok/s (5.48×)
```

**Key finding:** throughput scales near-linearly from conc=1 to conc=4 (3.36× for 4× concurrent requests) — the MI300X VF HBM bandwidth is not saturated at this model size. At conc=8, scaling efficiency drops to 68% — latency p50 rises to 3,093 ms, indicating the single-GPU KV-cache is filling and scheduler is batching at or near capacity.

---

## VRAM behaviour

| Condition | VRAM used (approx.) | Notes |
|-----------|--------------------|-|
| Model loaded, no requests | ~64 GiB | Qwen3-32B fp16 weights + activations |
| Under concurrency=8 load | ~80–90 GiB | KV-cache growth visible in `rocm-smi --showmeminfo vram` |
| Max observed | <120 GiB | Well within 191.69 GiB — no OOM |

**`rocm-smi` command used for live monitoring during run:**
```bash
watch -n 1 'rocm-smi --showmeminfo vram --showtemp --showpower'
```

**No OOM events** across all concurrency levels. The 191.69 GiB HBM3 gives significant headroom for KV-cache even at conc=8 with 128-token outputs for a 32B model.

---

## What failed / needed tuning

| Issue | Symptom | Resolution |
|-------|---------|------------|
| vLLM cold start | First request latency ~3× higher | Standard vLLM behaviour; model weights paged into HBM on first prefill |
| p99 TTFT spike (130 ms) | One outlier in TTFT measurements | Isolated to first request after idle period; subsequent requests stable at 62–84 ms |
| `amd-smi metric` unavailable | `amd-smi metric -m TEMPERATURE_JUNCTION` returned invalid-parameter | Used `rocm-smi --showtemp` instead — authoritative on this ROCm version |
| K8s GPU resource | `amd.com/gpu: 1` must be set in both `requests` and `limits` | GPU Operator does not expose partial allocation; the full VF is assigned |

---

## Reproduce

```bash
# From repo root, with ROCm PyTorch env active
pip install -r requirements.txt

# Run hard tier (concurrency sweep)
python3 src/infer_harness.py \
  --base-url https://airmui.<your-domain>/v1 \
  --model Qwen/Qwen3-32B \
  --tier hard \
  --out results/mi300x-vf-qwen3-32b-rocm641-$(date +%Y%m%d).json
```

**Stack BOM:**

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04.2 LTS |
| Kernel | 6.8.0-58-generic |
| ROCm | 6.4.1 |
| PyTorch | 2.4.1+rocm6.0 |
| GPU | AMD Instinct MI300X VF (gfx942) |
| VRAM | 191.69 GiB HBM3 |
| K8s | RKE2 v1.34.1+rke2r1 |
| AMD GPU Operator | v1.4.1 |
| Model | Qwen3-32B (fp16) |
| Serving | AIRM / vLLM (ROCm backend) |
