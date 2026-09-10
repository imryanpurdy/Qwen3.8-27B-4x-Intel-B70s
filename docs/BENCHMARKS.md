# Benchmarks — optimized Qwen3.8-27B-FP8 on 4× Intel Arc Pro B70

## Measured results (jobe, 2026-09-06, median of 5 runs, idle cooled box)

Final config: float16 + `VLLM_XPU_FP8_BLOCK_W8A16=1` + MTP5 + `max_model_len=262144`,
custom small-M kernel live (`xe2_small_m=True` in banner, op runtime-registered).

| Workload | tok/s | vs prior config (bf16/MTP8/9216) |
|---|---|---|
| decode, short prompt (64) → 512 out | **81.1** | +19% (and full 512 tokens — EOS no longer fires early) |
| decode, p512 → g128 | **46.1** | +18% |
| decode, p1024 → g256 | **45.1** | +11% |
| prefill E2E, 16k prompt | ~3452 | — |
| prefill E2E, 32k prompt | ~3447 | — |

Config history on this box:

| Config | short→512 | p512→g128 | p1024→g256 |
|---|---|---|---|
| stock, no MTP (original baseline) | ~16–17 | — | — |
| bf16 / MTP8 / max_model_len 9216 | 68.1* | 39.1 | 40.8 |
| float16 + W8A16 / MTP5 / 262144 (final) | **81.1** | **46.1** | **45.1** |

\* EOS fired at ~350 tokens on that path — not comparable wall-time (see discipline below).

## Honest framing (kept deliberately)

- These are absolute rates on the custom kernel stack. The kernel's edge over the
  stock path is ≈ **2× at small M** (the decode-critical regime, per the campaign's
  §3e note); earlier multi-hundred-percent figures were stale artifacts and are retracted.
- No clean head-to-head vs the interim bf16/MTP8 baseline was possible while both
  containers were up (TP4 uses ~25.75 GiB/card; the two configs are mutually
  exclusive on 4 cards). The deltas above are config-to-config on the same box.
- Rates from other repositories (e.g. 84.65 tok/s from the GPTQ G128 lane) use
  different workloads/quants — not comparable.

## Measurement discipline

1. **Never divide `completion_tokens` by wall clock.** Prefill dominates the wall
   and buries the decode edge. Isolate the decode phase from server usage counts /
   streaming chunk deltas (this repo's `bench/bench.sh` does that).
2. **Prefill rate** = `prompt_tokens / wall` with `max_tokens=1`.
3. **Thermal state matters.** B70s throttle under back-to-back load — a drift from
   103 → 43 tok/s was observed on this box within one hot campaign. Idle, cooled
   box; median of ≥ 3–5 runs; never compare a hot box to a cold one.
4. **`max_model_len` is enforced as prompt + output.** Read it live from
   `/v1/models` instead of trusting docs — this bit the campaign itself
   (docs claimed 262144 while the box ran 9216).
5. **Never compare wall-time across dtype paths.** The W8A16 path fires EOS
   earlier (~350 vs full 512 tokens in one config) — same wall-time ≠ same tokens.
6. **MTP acceptance** is visible in server logs
   (`SpecDecoding metrics: Mean acceptance length ... Per-position acceptance rate ...`)
   — the final config showed mean acceptance length ≈ 3.2–3.6, per-position
   0.79/0.58/0.48/0.45/0.34 on mixed workloads, and up to 5.9 (near-perfect) on
   repetitive prompts.
