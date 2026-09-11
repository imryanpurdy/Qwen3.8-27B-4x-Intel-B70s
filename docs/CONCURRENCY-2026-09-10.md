# Concurrency campaign 2026-09-10: seqs=4, the causal_conv1d wall, and its fix

## FINAL STATE (validated): one mode does both

`./start.sh` (MTP5 + MAX_NUM_SEQS=4 default) now serves concurrency AND keeps
speculative speed, via the **split-wrapper** fix in
`patched-sources/_xpu_ops.py`:

| Metric | Value |
|---|---|
| 4-stream MTP5 aggregate (4×2500 tok) | **205.1 tok/s** (51-54 each, zero failures) |
| vs old k=0 workaround | 205.1 vs 193.7 (acceptance gains kept) |
| Mixed-path canary (canaries batched against a live decode stream) | **5/5 bit-identical** vs eager baseline |
| Pure MTP path after fix (bench.sh) | 90.9 / **139.3** / 114.5 (−2-3% vs pre-fix 92.3/144.2/117.9, within thermal noise) |
| Prefill | unchanged (3698/3530) |
| Engine survival | alive after both volleys + mixed canary |

## The wall

With `--max-num-seqs 4` + MTP5, four parallel requests eventually share a step
where one request's prefill chunk coexists with another's MTP decode. The fused
`gdn_attention` op rejected that batch shape hard (`causal_conv1d does not
support spec-decode and non-spec tokens in the same invocation`) — worker death,
engine gone. Kernel-level, not schedulable around; `--no-enable-chunked-prefill`
is not a valid escape (vLLM warns it crashes this model class).

## The fix (validated): split-wrapper in _xpu_ops.py

Reading the kernel source (`kernel/gdn_attn_interface.cpp`, committed) showed:
1. The fused op is a thin chain of two split ops — `causal_conv1d` (→ {q,k,v,z,b,a}
   + conv_state update) and `gated_delta_rule` (→ core_attn_out + ssm_state update)
   — each accepting a single population per call, with the guard living only in
   the fused entry point.
2. Request cache slots never overlap across populations (a slot is in exactly one
   phase per scheduler step), so both populations can safely share conv_state /
   ssm_state across two back-to-back calls.

The wrapper now detects mixed batches (`num_spec_decodes > 0 and
num_prefills+num_decodes > 0`) and: gathers compact token copies per population
(token_indx order) → runs the spec population through causal_conv1d +
gated_delta_rule with identity spec indices → runs the non-spec population
likewise with the original prefill/decode metadata → `index_copy_` scatter of
outputs (core_attn_out and z) back to original token rows. Pure-population steps
take the original fused path byte-identical (canary-proven 5/5 before the
concurrency volley).

Validation sequence: static canary 5/5 on pure path → the exact volley that
previously killed the engine (clean, 205.1) → mixed-path canary 5/5 (the
decisive correctness proof) → comparable bench (−3% pure-path cost).

## History (superseded workarounds)

- `MTP_DEPTH=0 ./start.sh` (k=0 concurrency): 193.7 aggregate — kept as an env
  knob, no longer needed. Note: this runtime rejects `num_speculative_tokens=0`;
  start.sh omits the whole speculative-config flag.
- vLLM scheduler-side escapes: none valid for this model class (chunked prefill
  must stay on).

