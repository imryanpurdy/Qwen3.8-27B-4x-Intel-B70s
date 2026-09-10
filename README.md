# Qwen3.8-27B-FP8 on 4× Intel Arc Pro B70 — TP4 + MTP5 + custom small-M kernel

Optimized serving stack for **Qwen3.8-27B-FP8** on four Intel Arc Pro B70 32 GB cards:
vLLM XPU TP4, MTP5 speculative decoding, 262,144-token context, and a **custom
`xe2_block_fp8_small_m` CUTLASS kernel** for the small-M block-FP8 GEMMs that dominate
single-user decode on this model.

> **Status (2026-09-09):** the serving container is DOWN — the host rebooted on
> 2026-09-06 and wiped `/tmp`, which held the kernel artifacts and patch scripts.
> All recipe inputs were recovered and made durable (see `docs/RECOVERY-2026-09-09.md`).
> Relaunch: `./start.sh`.

## Measured (jobe, 2026-09-06, median of 5, idle cooled box)

| Workload | tok/s |
|---|---|
| decode, short prompt (64) → 512 out | **81.1** |
| decode, p512 → g128 | **46.1** |
| decode, p1024 → g256 | **45.1** |
| prefill E2E, 16k / 32k | ~3452 / ~3447 |
| context | 262,144 (was 9,216 in the earlier bf16 config) |

vs the prior config (bf16 / MTP8 / 9k ctx): **+11–19% decode**. Honest framing:
the custom kernel's edge over the stock path is ≈2× at small M (the decode-critical
regime), not the multi-hundred-percent figures from stale early artifacts. Numbers
from other repos (84.65 etc.) use different workloads — not comparable.

## What's in this repo

| Path | What | Provenance |
|---|---|---|
| `start.sh` / `stop.sh` | Durable relaunch/stop. Kernel artifacts and patched vLLM files are **bind-mounted over stock** — no startup patching, no idempotency traps, nothing in `/tmp` | authored 2026-09-09 from the verified live recipe |
| `.env.sample` | Every knob (image digest, container, port, artifact paths) | authored |
| `kernel/vxk_worktree.patch` | The custom kernel: 395 insertions across 8 files on `vllm-project/vllm-xpu-kernels@1796aa8` — `xe2_block_fp8_small_m` CUTLASS/CUTE device GEMM (block-FP8, small-M) + torch binding | true bytes from the build container's git worktree |
| `build/jobe_xpuc_fg.sh` | The original foreground build step | true bytes |
| `build/build-kernel.sh` | Full rebuild recipe (devel image, venv, oneAPI, `cmake --build . --target _xpu_C -j2`) | authored from verified invariants |
| `patches/fix_smallm_placement.py` | Repairs the template-header orphaning the original small-M insert produced | true bytes |
| `patches/diffs/*.patch` | The 5 vLLM-side patches, regenerated as exact diffs (stock `ac7509e2b` vs deployed files) | true bytes |
| `patched-sources/` | The post-patch vLLM files exactly as deployed | true bytes (dead container's writable layer) |
| `bench/bench.sh` | Median-of-5 E2E decode/prefill probe (server usage counts, no wall-clock division) | authored |
| `docs/BENCHMARKS.md` | Methodology + all measured numbers | |
| `docs/RECOVERY-2026-09-09.md` | The /tmp-wipe incident and how everything was recovered | |

## The five vLLM-side patches (what they actually do)

1. **`linear/scaled_mm/xpu.py`** — the W8A16 dispatcher: sets
   `apply_input_quant = False` (skip dynamic input quant), routes block-FP8 GEMM to
   `xe2_block_fp8_small_m` when `M ≤ 40` and K, N are 128-aligned (decode tiles),
   else to `fp8_gemm_w8a16`. Banner line `[B70_FP8_STACK] apply_input_quant=False use_w8a16=True xe2_small_m=True` comes from here.
2. **`_xpu_ops.py`** — fake (`register_fake`) schemas for `xe2_block_fp8_small_m` and
   `fp8_gemm_w8a16` so torch.compile/inductor can trace through them.
3. **`v1/executor/multiproc_executor.py`** — per-worker spawn-time
   `ZE_AFFINITY_MASK=<global_rank>` injection (each worker interpreter sees exactly one B70).
4. **`v1/worker/xpu_worker.py`** — worker-side affinity assertion (`B70_WORKER_AFFINITY=1`):
   rank must see exactly 1 device; `LOCAL_RANK=0` rewrite.
5. **`v1/worker/mamba_utils.py`** — **pointer-wrap fix**: XPU device pointers ≥ 2^63 are
   stored two's-complement-wrapped into int64 block-table / copy-spec tensors
   (`B70_PTR_WRAP_*`), because kernels add only small offsets to them.

Kernel artifacts are mounted over stock
`/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/`:
`_xpu_C.abi3.so` (96 MB), `libgrouped_gemm_xe_2.so`, `libgrouped_gemm_xe_default.so`.
They are NOT committed here — rebuild with `build/build-kernel.sh` or copy from
`jobe:~/xpu_artifacts/` (sha256s in `artifacts/README.md`).

## Quick start (on the B70 host)

```bash
# 0) one-time: kernel artifacts in ~/xpu_artifacts (see artifacts/README.md),
#    model at ~/models/Qwen3.8-27B-FP8 (~29 GB)
cp .env.sample .env   # edit if your paths differ

# 1) launch (durable: mounts ~/xpu_artifacts + this repo's patched-sources/)
./start.sh

# 2) verify
curl -s http://localhost:11438/v1/models        # "qwen38", max_model_len 262144
docker logs qwen28tp4m 2>&1 | grep B70_FP8_STACK # apply_input_quant=False use_w8a16=True xe2_small_m=True
docker exec qwen28tp4m python -c "import torch, vllm_xpu_kernels._xpu_C as m; print('small_m_op=', hasattr(torch.ops._xpu_C,'xe2_block_fp8_small_m'))"   # True (import first!)
```

Verified live contract (do not "fix" these — they are load-bearing):
- `VLLM_XPU_FP8_BLOCK_W8A16=1` + `--dtype float16`, or you land on the slower W8A8 path silently.
- Do NOT set `CCL_ATL_TRANSPORT=ofi` (prefill 3538 → 1132 tok/s on this box).
- MTP **5** (measured better than 8). `VLLM_XPU_ENABLE_XPU_GRAPH=1` + MTP = corrupted outputs — leave off.
- Base image ENTRYPOINT is `vllm`: always `--entrypoint bash` + `-lc "... exec vllm serve ..."`.
- One TP4 container at a time on the 4 cards (~25.75 GiB/card).
- Op verification: import `vllm_xpu_kernels._xpu_C` first, then `hasattr` — a bare `import torch` check returns False (runtime TORCH_LIBRARY registration; `nm -D` also lies).

## Bench

```bash
./bench/bench.sh 11438          # median of 5 per workload, E2E from server usage
```

Discipline: never `completion_tokens / wall` (prefill buries decode); B70s thermally
throttle under back-to-back runs — idle cooled box, median ≥ 3–5; `max_model_len` is
prompt+output — read it live from `/v1/models`; W8A16 fires EOS earlier — never
compare wall-time across dtype paths.

## Credits

- vLLM XPU + `vllm-project/vllm-xpu-kernels` (base `1796aa8`), vLLM `ac7509e2b` line
- steveseguin/b70-optimization-lab — the 4×B70 reference and adopted env/flag contract
- EdgeQuant — original stack assembly, patch authoring, and the measured campaign
