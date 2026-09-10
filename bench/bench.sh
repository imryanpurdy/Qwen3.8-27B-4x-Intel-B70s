#!/usr/bin/env bash
# ============================================================================
# bench.sh — decode/prefill probe for the optimized 27B stack.
#
# Methodology (matches the 2026-09-06 measurement campaign):
#   - decode rate isolates the decode phase: streaming request, rate =
#     (completion_tokens - 1) / (t_last_chunk - t_first_chunk). Prefill is NOT
#     in the denominator (dividing completion_tokens by wall clock buries the
#     decode edge under TTFT).
#   - prefill rate = prompt_tokens / wall with max_tokens=1.
#   - median of N runs (default 5), B70s thermally throttle under back-to-back
#     load — let the box idle/cool between campaigns; don't compare a hot box
#     to a cold one (drift 103 -> 43 tok/s has been observed on this hardware).
#   - max_model_len is enforced as prompt+output: read it live from /v1/models.
#
# Usage: ./bench.sh [PORT] [RUNS]    (defaults: 11438, 5)
# ============================================================================
set -euo pipefail
PORT="${1:-11438}"
RUNS="${2:-5}"
BASE="http://localhost:$PORT"

command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 || { echo "python required"; exit 1; }
PY=$(command -v python3 || command -v python)

curl -fsS -m 10 "$BASE/v1/models" >/dev/null || { echo "server not answering on :$PORT"; exit 1; }
echo "server up. reading max_model_len..."
$PY - "$BASE" <<'PYEOF'
import json, sys, urllib.request
d = json.load(urllib.request.urlopen(sys.argv[1] + "/v1/models"))
print("max_model_len =", d["data"][0].get("max_model_len"))
PYEOF

run_case() {  # name prompt_tokens gen_tokens
    local name="$1" pt="$2" gt="$3"
    echo "=== $name (p$pt -> g$gt), $RUNS runs ==="
    $PY - "$BASE" "$pt" "$gt" "$RUNS" <<'PYEOF'
import json, sys, time, urllib.request, statistics
base, pt, gt, runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
prompt = ("Explain the operational considerations for serving a large mixture-of-experts "
          "model on multiple consumer GPUs. Cover memory placement, kernel selection, "
          "speculative decoding acceptance, and thermal behavior. ") * max(1, pt // 48)
rates = []
for i in range(runs):
    body = json.dumps({"model": "qwen38", "prompt": prompt, "max_tokens": gt,
                       "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; toks = 0; toks_final = None
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            d = json.loads(payload)
            ch = (d.get("choices") or [{}])[0]
            text = ch.get("text") or ""
            if text and first is None:
                first = time.perf_counter()
            if first is not None and text:
                last = time.perf_counter(); toks += 1
            u = d.get("usage")
            if u and u.get("completion_tokens"):
                toks_final = u["completion_tokens"]
    if toks_final is None: toks_final = toks
    if first and last and last > first:
        rates.append((toks_final - 1) / (last - first))
print(f"  decode tok/s runs: {[round(r, 1) for r in rates]}")
print(f"  MEDIAN: {statistics.median(rates):.1f} tok/s")
PYEOF
}

run_prefill() {  # name prompt_tokens
    local name="$1" pt="$2"
    echo "=== $name (prefill p$pt, g1) ==="
    $PY - "$BASE" "$pt" "$RUNS" <<'PYEOF'
import json, sys, time, urllib.request, statistics
base, pt, runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
prompt = ("Summarize the following operational runbook. " + 
          "The serving stack must handle memory placement across four accelerators, "
          "kernel dispatch for block-scaled floating point GEMMs, and scheduler "
          "interaction with speculative draft heads. ") * max(1, pt // 40)
rates = []
for i in range(runs):
    body = json.dumps({"model": "qwen38", "prompt": prompt, "max_tokens": 1,
                       "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(base + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        json.loads(r.read())
    rates.append(pt / (time.perf_counter() - t0))
print(f"  prefill tok/s runs: {[round(r, 1) for r in rates]}")
print(f"  MEDIAN: {statistics.median(rates):.1f} tok/s")
PYEOF
}

run_case "short->512"  64  512
run_case "p512->g128"  512 128
run_case "p1024->g256" 1024 256
run_prefill "prefill-16k" 16384
run_prefill "prefill-32k" 32768
echo ""
echo "Reference (2026-09-06, idle cooled box, median of 5):"
echo "  81.1 short->512 | 46.1 p512->g128 | 45.1 p1024->g256 | prefill ~3452/3447"
