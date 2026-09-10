# Campaign 2026-09-10: graphs unlocked, 144 tok/s prose decode, bit-exact

## Headline

`VLLM_XPU_ENABLE_XPU_GRAPH=1` + MTP5 on the 4×B70 27B stack:
**144.2 tok/s p512→g128 prose decode, canary 5/5 bit-identical vs eager.**
The long-standing "XPU graphs + MTP corrupt outputs / can't run on TP4" contract
does not reproduce on this runtime (v0.27.2rc1.dev77+gac7509e2b, image
`f01e24f6`). Prefill unchanged (piecewise path), engine step ~73ms → ~26ms,
bandwidth utilization 18% → ~50% of the 2.4 TB/s aggregate ceiling.

## Full decision matrix (median of 5, idle cooled box)

| Config | short prose→512 | p512→g128 | p1024→g256 | count-probe (420) | canary |
|---|---|---|---|---|---|
| eager k=5 (pre-campaign) | 40.2 | 59.8 | 49.5 | 83.6 | baseline |
| eager k=3 | 36.9 | 49.0 | 45.2 | 60.3 | — |
| **graphs k=5 (shipped default)** | **92.3** | **144.2** | **117.9** | 192.9 | **5/5 identical** |
| graphs k=7 | 82.5 | 151.2 | 112.1 | **222.4** | 5/5 identical |

- k=3 verdict: loses to k=5 on this stack even in eager (draft passes are cheap —
  the MTP head rides the small-M kernel path). The win was never in shortening
  the draft loop; it was the verify step + collectives.
- k=7 verdict: specialist for high-acceptance content (acceptance 6.73/7,
  222 tok/s) but loses prose to k=5 (5th–7th draft positions wasted).
- Graph capture observed: FULL decode graph (size 1) + PIECEWISE
  mixed-prefill-decode (1–4096), ~78s one-time compile at boot. GMU 0.85 fits:
  7.65 GiB weights + 0.15 GiB graph + 17.28 GiB KV per rank.

## Canary methodology

5 deterministic prompts (factual / code / reasoning / repeat / needle),
temperature 0, max_tokens 64, sha256 of the 64-token completion.
Files: jobe `~/qwen28/canary/canary_eager_k5.txt` (baseline),
`canary_graph_k5.txt`, k7 hashes in this campaign log. Every graphs config MUST
re-run the canary before being trusted — corruption is config-specific.

## What did NOT need to be built (and why that matters)

The plan had ranked: (1) custom whole-graph capture for TP4 — the "custom
cudagraph for XPU" build; (2) fused draft-loop megakernel; (3) TP2.
The runtime's own graph path turned out to work on TP4 with MTP on this build —
the lab-era negatives were properties of the August nightly (`e9d1398d9`,
torch 2.13), not of XPU graphs generally. Lesson recorded: re-probe quarantined
negatives after every runtime bump; the quarantine list does not transfer across
builds.

## Remaining headroom (for the next campaign)

- ~26ms/step is still ~2× the 13ms bandwidth floor: capture sizes cap verify
  batch at 8 tokens ([1,2,4,8] — k=7 max, no concurrency capture), and the draft
  loop still host-orchestrates inside the step.
- Next levers: draft-loop device-side fusion (the megakernel — kills ~10ms/step),
  capture-size extension for concurrency, TP2-vs-TP4 re-check under graphs.
