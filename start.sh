#!/usr/bin/env bash
# ============================================================================
# start.sh — relaunch the optimized Qwen3.8-27B-FP8 stack on 4x Intel Arc B70
#
# Durability redesign vs the original 2026-09-05 launch:
#   - kernel artifacts mounted from ~/xpu_artifacts (NOT /tmp — that died in
#     the 2026-09-06 reboot; see docs/RECOVERY-2026-09-09.md)
#   - patched vLLM files bind-mounted over stock from this repo's
#     patched-sources/ — no startup patch scripts, no idempotency crash-loops
#   - docker rm -f + fresh create (docker restart re-runs the entrypoint on an
#     already-patched FS and crash-loops)
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$SCRIPT_DIR/.env" ]] && source "$SCRIPT_DIR/.env"

IMAGE="${IMAGE:-vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f}"
CONTAINER="${CONTAINER:-qwen28tp4m}"
PORT="${PORT:-11438}"
ARTIFACTS="${ARTIFACTS:-$HOME/xpu_artifacts}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-FP8}"
SOURCES="${SOURCES:-$SCRIPT_DIR/patched-sources}"

err() { echo "[ERR ] $*" >&2; exit 1; }
ok()  { echo "[ OK ] $*"; }

# ---- preflight -------------------------------------------------------------
for f in _xpu_C.abi3.so libgrouped_gemm_xe_2.so libgrouped_gemm_xe_default.so; do
    [[ -f "$ARTIFACTS/$f" ]] || err "missing $ARTIFACTS/$f — build with build/build-kernel.sh (see artifacts/README.md)"
done
[[ -d "$MODEL" ]] || err "model dir missing: $MODEL"
[[ -f "$SOURCES/_xpu_ops.py" ]] || err "patched-sources missing under $SOURCES"
docker info >/dev/null 2>&1 || err "docker daemon unreachable"
ok "preflight: artifacts + model + docker present"

# ---- clean slate -----------------------------------------------------------
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# ---- durable mounts over stock files ---------------------------------------
K=opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
V=opt/venv/lib/python3.12/site-packages/vllm
MOUNTS=(
  -v "$ARTIFACTS/_xpu_C.abi3.so:/$K/_xpu_C.abi3.so:ro"
  -v "$ARTIFACTS/libgrouped_gemm_xe_2.so:/$K/libgrouped_gemm_xe_2.so:ro"
  -v "$ARTIFACTS/libgrouped_gemm_xe_default.so:/$K/libgrouped_gemm_xe_default.so:ro"
  -v "$SOURCES/_xpu_ops.py:/$V/_xpu_ops.py:ro"
  -v "$SOURCES/v1__worker__xpu_worker.py:/$V/v1/worker/xpu_worker.py:ro"
  -v "$SOURCES/v1__worker__mamba_utils.py:/$V/v1/worker/mamba_utils.py:ro"
  -v "$SOURCES/v1__executor__multiproc_executor.py:/$V/v1/executor/multiproc_executor.py:ro"
  -v "$SOURCES/model_executor__kernels__linear__scaled_mm__xpu.py:/$V/model_executor/kernels/linear/scaled_mm/xpu.py:ro"
)

# ---- launch ----------------------------------------------------------------
# ENTRYPOINT gotcha: base image ENTRYPOINT is `vllm` — must use --entrypoint bash
docker run -d --name "$CONTAINER" \
  --entrypoint bash \
  --device /dev/dri -v /dev/dri:/dev/dri \
  --group-add 991 --ipc=host --shm-size 64m \
  -p "$PORT:8000" \
  -v "$MODEL:/model:ro" \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e VLLM_XPU_FP8_BLOCK_W8A16=1 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e B70_WORKER_AFFINITY=1 \
  -e ZE_AFFINITY_MASK=0,1,2,3 \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_ALLTOALL_TMP_BUF=1 \
  -e "LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels:/opt/venv/lib:/usr/lib" \
  "${MOUNTS[@]}" \
  "$IMAGE" \
  -lc "exec vllm serve /model --quantization fp8 --dtype float16 --tensor-parallel-size 4 --max-model-len 262144 --max-num-seqs 1 --async-scheduling --block-size 64 --mamba-ssm-cache-dtype float16 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --language-model-only --port 8000 --served-model-name qwen38 --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_DEPTH:-5}}'"

ok "container $CONTAINER starting on :$PORT (weight load takes ~4 min)"
ok "follow:  docker logs -f $CONTAINER"
ok "verify:  curl -s http://localhost:$PORT/v1/models"
