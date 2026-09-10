# Kernel artifacts — provenance and checksums

The three custom-built binaries are NOT committed (96 MB + 8 MB). They live on the
serving host at `~/xpu_artifacts/` (durable — moved off `/tmp` after the 2026-09-06
reboot wipe).

| File | Size | sha256 |
|---|---|---|
| `_xpu_C.abi3.so` | 95,865,024 B | `593a7107d3d20304f3d37b7ea20ca00b2eea361fff2aad5c1107fd91c5abc43b` |
| `libgrouped_gemm_xe_2.so` | 6,644,472 B | `2da4a494d4014e58e8a4a91e84cec39a22143982c6f847188a2e5e7e94f90d13` |
| `libgrouped_gemm_xe_default.so` | 1,247,176 B | `4b1ca1e660b80bc422dcc3bc7bd71f9e471615a2baf1f012b340593c62ab5161` |

Provenance: built 2026-09-05 in the `vxkbuild` container (devel image
`intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04`, venv at `/vxk-venv`,
source at `/vxk` = vllm-project/vllm-xpu-kernels @ `1796aa8` + `kernel/vxk_worktree.patch`
+ `patches/fix_smallm_placement.py`). Recovered 2026-09-09 from the stopped
`vxkbuild` container's filesystem (`docker cp vxkbuild:/vxk/build/...`) — the
original `/tmp/xpu_artifacts/` copies were destroyed by the reboot.

Verify the small-M op is really in the artifact:
```bash
strings _xpu_C.abi3.so | grep xe2_block_fp8_small_m    # symbol present
# definitive runtime check (after serving starts):
docker exec qwen28tp4m python -c "import torch, vllm_xpu_kernels._xpu_C; print(hasattr(torch.ops._xpu_C,'xe2_block_fp8_small_m'))"
```
