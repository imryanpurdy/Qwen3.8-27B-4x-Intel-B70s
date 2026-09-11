# Concurrency campaign 2026-09-10: seqs=4, the causal_conv1d wall, and the two-mode serving model

## Result matrix (all graphs ON, canary-verified where MTP is on)

| Mode | Launch | Single-stream | 4-stream aggregate | Notes |
|---|---|---|---|---|
| **MTP5** (default) | `./start.sh` | **144.2** prose / **222.4** count-class (k=7) | ❌ crashes | fastest for one user |
| **k=0 concurrency** | `MTP_DEPTH=0 ./start.sh` | ~72 (lab-measured class) | **193.7** (4×2500 tok, 48.4 each, fair, stable) | fastest for many users |

## The causal_conv1d wall (why MTP + concurrency crashes)

With `--max-num-seqs 4` + MTP5, four parallel requests eventually share a step
where one request's prefill chunk coexists with another's MTP decode. The GDN
kernel in `_xpu_C` rejects that batch shape hard:

```
RuntimeError: causal_conv1d does not support spec-decode and non-spec (prefill +
decode) tokens in the same invocation; the spec path and the non-spec path are
mutually exclusive
```

Worker death, engine gone (all four client streams got HTTP 500). This is a
kernel-level constraint, not a scheduler config — `--no-enable-chunked-prefill`
is NOT a valid escape (vLLM explicitly warns manual disabling may crash this
model class; `MTP_DEPTH=0` is the clean one).

## The fix that exists (built, pending validation): the split-wrapper

`_xpu_ops._gdn_attention_core_xpu_impl` (bind-mounted over stock by `start.sh`)
passes spec + non-spec tokens into ONE `torch.ops._xpu_C.gdn_attention` call.
The kernel can serve each population fine *separately* (both single-population
paths are exercised daily). So: slice the batch into the two groups using the
metadata the wrapper already receives (`spec_query_start_loc`,
`non_spec_query_start_loc`, `spec_state_indices_tensor`,
`non_spec_state_indices_tensor`), invoke the kernel twice, write both outputs
into `core_attn_out` at their original token indices. All two calls stay
single-population. Cost: one extra kernel launch per mixed step (~1ms).
Open invariant to verify on the rig: conv/ssm state update ordering across the
two invocations (non-spec must not clobber spec rows' cache lines) — needs a
canary + long-run before trust. File: `docs/` this note is the spec.

## Two-mode operation (what start.sh ships now)

- One user at a time (chat, agent, benchmark): default `./start.sh` → MTP5, 144.
- Burst/multi-user: `MTP_DEPTH=0 ./start.sh` → 193.7 aggregate at 4 streams.
- Flipping between modes = one env var + a ~4.5 min relaunch.
- `MAX_NUM_SEQS` env scales the bucket plan ([8,16,32] at k=0; [8,12,18,24,36]
  at k=5) — higher seq counts need proportionally larger capture buckets.

## Canary discipline held

The seqs=4+MTP5 boot was canary 5/5 text-identical before the concurrency probe
(the earlier "0/5" was a comparison-script bug, hashes were correct all along).
Crashes only occur at runtime on mixed batches — static canaries can't catch
them; the 4-parallel-stream volley is now part of the standard acceptance run.
