#!/usr/bin/env bash
# ============================================================================
# build-kernel.sh — rebuild the custom vllm-xpu-kernels _xpu_C artifact set
# from source. Run INSIDE the devel container (see commands at bottom).
#
# Inputs (all true bytes, committed in this repo):
#   kernel/vxk_worktree.patch   — the 395-line custom-kernel diff
#   patches/fix_smallm_placement.py — repairs template-header orphaning
#
# Verified invariants (do not change without re-measuring):
#   - build in the DEVEL image (intel/deep-learning-essentials:2026.0.0-devel):
#     needs icpx + torch 2.13.0+xpu with the EXACT ABI of the runtime image.
#   - patch order is load-bearing: apply vxk_worktree.patch, THEN
#     fix_smallm_placement.py (the raw insert orphans xe_gemm_4bits' template
#     header; the fix swaps the block before the orphaned header). Skipping the
#     fix = compile failure.
#   - target artifact is _xpu_C.abi3.so (NOT _C). Build: cmake --build . --target _xpu_C -j2
#   - devel-built RUNPATH points at a path that doesn't exist in the runtime
#     image -> ImportError is swallowed -> "Failed to infer device type" at
#     arg-parse. Fix at serve time: LD_LIBRARY_PATH including the kernels dir
#     (start.sh sets it), or patchelf --set-rpath on the artifacts.
#   - verify the op AFTER importing the module:
#       import torch, vllm_xpu_kernels._xpu_C
#       hasattr(torch.ops._xpu_C, 'xe2_block_fp8_small_m')   # must be True
#     (runtime TORCH_LIBRARY registration; nm -D and bare-import probes lie.)
#
# Original artifacts (if you just want to re-deploy instead of rebuild):
#   jobe:~/xpu_artifacts/  sha256:
#     _xpu_C.abi3.so               593a7107d3d20304f3d37b7ea20ca00b2eea361fff2aad5c1107fd91c5abc43b
#     libgrouped_gemm_xe_2.so      2da4a494d4014e58e8a4a91e84cec39a22143982c6f847188a2e5e7e94f90d13
#     libgrouped_gemm_xe_default.so 4b1ca1e660b80bc422dcc3bc7bd71f9e471615a2baf1f012b340593c62ab5161
# ============================================================================
set -euo pipefail
: "${VXK_REF:=1796aa8bc8db4ac68d9cd19636cef88f3af81d2b}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

git clone https://github.com/vllm-project/vllm-xpu-kernels.git /vxk
cd /vxk && git checkout "$VXK_REF"
git apply "$REPO_DIR/kernel/vxk_worktree.patch"
VXK_SRC=/vxk/src python3 "$REPO_DIR/patches/fix_smallm_placement.py"

# venv with the runtime-matching torch (ABI must match the runtime image's torch 2.13.0+xpu)
python3 -m venv /vxk-venv && /vxk-venv/bin/pip install --upgrade pip
/vxk-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/xpu
source /opt/intel/oneapi/setvars.sh
cmake -S /vxk -B /vxk/build -DCMAKE_BUILD_TYPE=Release
cmake --build /vxk/build --target _xpu_C -j2     # original build used -j2 (RAM pressure)

mkdir -p /vxk/out
cp /vxk/build/_xpu_C.abi3.so /vxk/out/
cp /vxk/build/libgrouped_gemm_xe_2.so /vxk/out/
cp /vxk/build/libgrouped_gemm_xe_default.so /vxk/out/
sha256sum /vxk/out/*
echo "Deploy: copy /vxk/out/* to ~/xpu_artifacts/ on the serving host."
