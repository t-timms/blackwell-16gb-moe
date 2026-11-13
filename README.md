# Fitting a 35B MoE into 16 GB: REAP pruning + NVFP4 on consumer Blackwell

This is the pipeline I use to take a 35B-parameter mixture-of-experts model that ships at
65-71 GB in bf16 and turn it into something that loads, serves, and scores on an RTX 5070 Ti
with 16 GB of VRAM. It has produced three public releases: two 35B coders
([KAT-Coder-V2.5-Dev REAP-50 NVFP4](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4A16)
and [Ornith-1.5-35B-A3B REAP-50 NVFP4](https://huggingface.co/Ttimms/Ornith-1.5-35B-A3B-REAP-50-NVFP4A16)),
plus a [W4A4 build of the smaller ZAYA1-8B](https://huggingface.co/Ttimms/zaya1-8b-nvfp4-w4a4-uniform)
that shares the quantization and evaluation half. A large share of the elapsed time on each one
went to failures that exited cleanly and produced valid-looking artifacts that were wrong.

The write-up is organised as the pipeline runs: decide the compression budget, prune experts,
strip the parts that do not serve the target, quantize to NVFP4, get it running on vLLM, then
evaluate it so the number survives contact with a second run. Every number here is measured on
the released checkpoint unless it says otherwise.

## 1. The target is a hardware class, not a leaderboard

The goal is **SOTA within the 16 GB consumer / Blackwell class** -- an RTX 5070 Ti, compute
capability 12.0 (SM120), FP4-native tensor cores -- and nothing larger. A solo operator on one
consumer card is not going to beat a frontier model on unconstrained hardware, and framing the
work that way sets up a loss.

So the comparison set for any quality claim is fixed: **other quants and prunes of the same
base model that also fit in 16 GB.** Not the unquantized base (it does not fit, so retention
against it cannot even be measured on this hardware), and not GPT-class models. When I check
whether a build is any good, I run the identical eval on every entry in that specific set, in
the same session, with repeated draws where sampling is stochastic.

For Ornith, that comparison set was empty -- no pruned-and-quantized build of that base existed
anywhere at release time, at any size, as far as an exhaustive search could tell. For KAT-Coder
a REAP GGUF existed, but not a vLLM-servable 16 GB build with SWE-bench / HumanEval+ / MBPP+
numbers and their confidence intervals attached. That second thing is the actual claim: not
"better than the alternatives" but "the usable-and-measured version did not exist yet."

## 2. Pruning is not optional, and the arithmetic says how much

Quantization alone does not get a 35B MoE into 16 GB. Concretely, for KAT-Coder-V2.5-Dev:

| variant | size | fits 16 GB |
|---|---:|:---:|
| bf16 base | 69.3 GB | no |
| NVFP4, unpruned | 21.9 GB | no |
| REAP 25% + NVFP4 | ~16-17 GB | no, once KV cache is counted |
| **REAP 50% + NVFP4** | **12.45 GiB** | **yes**, with ~3 GB left for KV cache |

Ornith-1.5-35B-A3B is the same story: its vendor publishes several quant variants, and I
measured every one directly rather than trusting the README. The smallest is a GGUF Q4_K_M at
**21.71 GiB** -- still 5.7 GB over a 16 GB card and further over a realistic serving budget.
There is no quantization-only path for serving with everything in VRAM, which is what vLLM
does. For single-user llama.cpp, the MoE expert cache changes this answer. See section 10.

50% expert pruning is the ceiling I will run without a recovery step. Published results above
that start showing measurable long-horizon-editing degradation, and it is not safe to assume
from the architecture tag: REAP's own authors report Moonlight-16B-A3B collapsing to ~17%
accuracy recovery at 50% sparsity, while a published 50% prune of this base's size-class cousin
(Qwen3.6-35B-A3B, arXiv 2607.16721) showed no statistically detectable loss on its code
benchmark. Some bases prune well and some do not.

## 3. REAP: removing experts without a retrain

REAP (Router-weighted Expert Activation Pruning) scores each expert by how much it actually
contributes over a calibration set, drops the bottom half, and rescales the router over the
survivors. For a `qwen3_5_moe` model that is 256 experts down to 128. No gradient step, no
distillation. The Ornith prune ran in about 48 minutes on this card.

Two things bite here.

**The router-renormalization fix.** After dropping half the experts, the retained experts'
router weights have to be renormalized so their combined contribution compensates for the ones
removed. The upstream REAP adapter I started from had this **silently disabled** for the
`qwen3_5_moe` architecture -- a "successful" prune run with that bug produces a checkpoint that
looks fine and is quietly worse. The fix is small; it is committed to a fork for upstreaming,
and by the second project it was architecture-detected rather than hardcoded.

**Calibration residency and hybrid architectures.** The observe-and-prune pass needs the
model's weights somewhere. The options are: pin the whole thing in host RAM (`cpu_full`,
deterministic, needs ~65 GB resident plus headroom), keep it on GPU (`gpu_full`, does not fit),
or stream it block by block with disk offload (`layerwise`). On a homogeneous transformer,
`layerwise` is fine. On a hybrid Gated-DeltaNet + attention MoE like Ornith it surfaced three
real bugs in the pruning tool: one was an attention-mask assumption that only breaks when
`full_attention` and `linear_attention` blocks are interleaved, and the other two were the same
shape -- code reading or mutating expert weights *after* the block's `forward()` returned, by
which point `accelerate`'s offload hook had already moved those weights back to the meta
device. Two of the three were fixable; the third was one call site deeper. The right answer was
to re-measure the RAM headroom I had been too conservative about (the real footprint was
~65 GB, not the ~72 GB I estimated from the parameter count) and revert to `cpu_full`, which
had completed the whole pipeline cleanly, twice. Residency mode does not affect the resulting
model, so this was a pure risk/effort call, not a compromise.

## 4. What else comes off: vision towers and MTP heads

Modern open bases carry parts a text-only 16 GB coding model does not need. Removing them is
free VRAM, but only if you check what you are removing.

**Vision tower: phantom or trained?** KAT-Coder declares a vision tower in its config and ships
**no weights** for it -- transformers would materialise 333 untrained tensors (0.83 GiB) at
load time for nothing. Free to strip. Ornith's vision tower looked similar on paper, so I
checked it the same way: standard weight init leaves fingerprints that training erases.
LayerNorm affine weights start as an exact all-ones vector; `nn.Linear` biases start at exact
zero; `nn.Linear` weights start i.i.d. `N(0, 0.02^2)` with no depth structure. I loaded
Ornith's `model.visual.*` tensors straight from the safetensors shards and computed per-tensor
stats against the known-trained text backbone as a positive control. The vision norm weights
were **0-2.6% exactly 1.0** (an untrained module is ~100%), with real cross-layer structure,
and patch-embed biases spread across roughly [-3, +3]. Ornith's vision tower is **genuinely
trained.** Stripping it for a text-only build is still correct, but it is documented as
discarding a real capability for scope reasons, not removing dead weight -- anyone who wants it
back has to start from the upstream checkpoint.

**MTP head: a reload trap.** Ornith ships a Multi-Token-Prediction head, and it is not a
lightweight dense draft layer -- it is a complete extra decoder layer with its own 256-expert
MoE, router, shared expert, and attention block. The trap: `text_config.num_experts` is a
**single config field** shared by the backbone and the MTP block. Prune the backbone via the
stock CLI and that field gets rewritten to 128, while the MTP block's on-disk router weight is
still shape `(256, hidden)`. On the *next* load -- quantization, eval, or serving -- a size
mismatch on that present key is a hard `RuntimeError`, not a warning: a multi-hour prune run
that succeeds by its own logs and then breaks at the very next step. The fix for v1 is to
disable the head (`mtp_num_hidden_layers: 0`, drop any `mtp.*` tensors, reload-verify with a
real forward pass). What actually happened on the real checkpoint was a *different* failure
than I predicted -- the pruning tool never saved the MTP weights at all, leaving a config that
claimed a head that was not there -- but it lands in the same place, and a small-scale reload
check caught it right after the prune, before the multi-hour quantization step.

## 5. NVFP4 on consumer Blackwell, and where it fights you

NVFP4 is a block-structured 4-bit float: 16-element blocks, an FP8 (E4M3) scale per block, and
one FP32 global scale per tensor. On Blackwell it maps to the FP4 tensor cores. Several failure
modes are worth knowing before you start.

**The global-scale convention is silent when wrong.** `global_scale` must be `2688 / max_abs`
in divisor form, and the per-block FP8 scale must be pre-multiplied by it. Get the direction
backwards and the checkpoint **loads without error** and produces NaN or zero logits, which
downstream shows up as the model emitting nothing but pad tokens. There is no exception to
catch. This is the single most expensive mistake in the format.

**SM120 is not SM100.** Datacenter Blackwell (SM100) and consumer Blackwell (SM120) share a
name and not much else in the FP4 kernel path -- SM120 uses SM80-era `mma.sync` instructions
rather than SM100's `tcgen05.mma`, so kernels written for SM100 do not transfer, and
`flashinfer` ships no FP4 cubins for `sm_120` at all. The kernels that make weight-only NVFP4
work on consumer Blackwell (`cutlass_scaled_fp4_mm_sm120a`, `cutlass_fp4_group_mm`) exist in
the vLLM source tree but are **absent from the pre-built wheels**, so a source build with
`TORCH_CUDA_ARCH_LIST=12.0` is mandatory, not optional.

**The NVFP4 MoE path (W4A4 + FusedMoE) is still broken upstream on SM120.** This is the harder
case and it did not resolve on its own. On the ZAYA1-8B W4A4 build, three vLLM MoE backends
produce three *different* garbage outputs from bit-identical weights -- semantically different,
so at least one and probably more are numerically wrong. Getting coherent output at all
required a local rewrite of the W4A4 MoE method into an on-the-fly Python dequant path. It is
tracked across several open vLLM issues (#33333, #38971, #31085, #24968, #30694) and is the
reason the ZAYA1 W4A4 checkpoint needs a pinned older stack (vLLM 0.20.2, transformers 4.57.1)
and does not load on a current one. Weight-only NVFP4 (W4A16) on a non-MoE-fused path is fine;
W4A4 FusedMoE on this card is not, today.

**CUDA graph capture can corrupt output, and it is build-specific.** On that same ZAYA1 W4A4
build, enabling CUDA graphs produced fluent text unrelated to the prompt -- silently, under
greedy decoding, across three unrelated backends (`flashinfer_cutlass`, `cutlass`, and
weight-only `marlin`), all fixed by `enforce_eager=True`. But the KAT-Coder NVFP4A16 build runs
**correctly** with CUDA graphs in `PIECEWISE` mode on the same card -- it is weight-only, the
GEMM runs in bf16 via the Marlin kernel, and no FP4 arithmetic is on the graph. "SM120 CUDA
graphs are broken" is too strong; the accurate statement is that I have seen the FP4-arithmetic
path break under graph capture on SM120, so you verify per build by checking real output rather
than assuming either way.

**Weight-only (W4A16) vs weight-and-activation (W4A4).** W4A16 quantizes weights to 4-bit and
runs the GEMM in bf16. Weight-only is *expected to win at batch-1 by design*: decode there is
not compute-bound, so quantizing activations adds dequantization work without relieving the
real bottleneck (per-step dispatch overhead, per a roofline check). W4A4's wins are footprint
and batched throughput -- the ZAYA1 uniform build is 6.02 GB and scales to 96-98% of ideal at
batch-8. Pick based on whether you are latency-bound or memory/throughput-bound.

**Upstream gaps you patch locally, and which ones come back.** Getting the Ornith NVFP4
checkpoint to serve took three patches: an llm-compressor 2D-load gap that silently dropped
every quantized expert weight (addressed in llm-compressor PR #3080, which names an Ornith
checkpoint), a missing vLLM model-registry entry for the text-only class, and three
mamba-state classmethods the hybrid `IsHybrid` path requires that crashed on the first request
-- after model load and KV init succeeded, which is the worst place for a crash to land. The
two vLLM-side gaps are both fixed upstream now (vLLM #50210). That is the good case. The SM120
W4A4-MoE situation above is the bad case: same "patch locally, file it" move, still not
resolved upstream a release later.

## 6. Serving on stock vLLM

Once the checkpoint is right, serving is mostly flags, and a few are not obvious:

- **`--language-model-only`** (KAT): the base declares a vision tower, and without this flag
  vLLM profiles a 16K-token image budget through it and wastes the VRAM.
- **`--enable-prefix-caching`**: the single largest agentic lever I measured -- 45x on replayed
  conversation history (0.21 s vs 30.74 s for a 13K-token prefix). But on hybrid GDN/Mamba
  models, prefix caching does nothing below the ~528-token cache block boundary (a 479-token
  prompt gets 0% hits, 552 tokens gets ~95%), so early agent turns see no benefit.
- **`--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`**: the safe graph mode for the
  weight-only build (see section 5).
- **`--max-cudagraph-capture-size`** below its default of 512: vLLM's own Qwen3.5 recipe
  documents this to avoid a mamba-cache assertion on that serving path.
- Never set `--max-model-len` right at a measured KV-cache ceiling -- available KV cache swings
  ~0.5-1.4 GiB depending on what else is using the card (a desktop compositor, a browser).

## 7. Evaluating it so the number means something

This is where most of the silent failures live.

**Loglikelihood tasks measure ranking, not production.** HellaSwag, ARC, PIQA and friends score
pre-written continuations. A checkpoint that could not form a coherent sentence once scored
**61% on HellaSwag** in this project. If you are checking whether a quantization broke the
model, you need at least one **generative** benchmark (HumanEval, GSM8K) in the mix.

**The harness is nondeterministic even at greedy.** Five runs of `humaneval_instruct` on one
fixed W4A4 checkpoint, greedy decoding, fixed seed, ranged from **92.07 to 96.34** pass rate,
median 93.90 -- a 4.3-point spread with nothing changed between runs (nondeterminism in the
serving/eval stack: batch scheduling, kernel reductions). A single-draw comparison of two
quantizations inside that spread is measuring noise. The fix is 3-5 draws per condition, report
the median and the range, and treat differences smaller than the observed spread as unresolved.

**Version the protocol.** The Bible-Assistant project runs a sha256-pinned question suite with
a manifest carrying a `protocol_id`; any change to the questions, the judge rubric, or a
metric's meaning bumps the manifest to a new id, so scores stay comparable across time and a
run against the wrong suite fails fast instead of producing a number that looks valid. It also
caught its own metric being bad: three close checkpoints landed within 0.008 of each other on a
fuzzy-overlap metric -- inside that metric's noise floor -- which is what prompted adding a
cross-encoder semantic score built specifically to separate them.

**Use paired tests.** Comparing two quantizations of one base on the same items is a paired
design. An earlier "the difference is not measurable" claim in the ZAYA1 work came from an
underpowered *unpaired* test; the paired exact-binomial McNemar over 14,319 items bounded the
real cost of removing all mixed-precision exemptions at **-0.71 pp HellaSwag, 95% CI
[-1.26, -0.15]** -- small, but not zero, and only visible once the pairing was kept.

**Know which numbers are contaminated.** GSM8K is heavily contaminated; decontamination work
shows up to 13 pp of the score is memorised. Read it as a regression check ("arithmetic still
works"), not a capability claim, and say so.

**Exit code 0 means nothing.** Three separate failures in the ZAYA1 work -- CUDA graph
corruption, a calibration bug that produced a 977-row corpus containing 3 unique rows (one
repeated 972 times), and a missing `max_model_len` that truncated generations by context
instead of budget -- each exited cleanly and produced artifacts that passed every structural
check (right file size, valid JSON, well-formed manifest). Only inspecting the actual output at
a level where failure modes are distinguishable caught any of them: token ids and
`finish_reason`, not "the text looks empty."

## 8. Results, and what is actually distinct

| Build | Size | HumanEval+ | MBPP+ | SWE-bench Verified | Serving |
|---|---:|---:|---:|---:|---|
| [KAT-Coder REAP-50 NVFP4A16](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4A16) | 12.45 GiB | 89.0% [83.3, 92.9] | 90.5% [87.1, 93.0] | 26/50 = 52.0% | vLLM, CUDA graphs (PIECEWISE) |
| [Ornith-1.5 REAP-50 NVFP4A16](https://huggingface.co/Ttimms/Ornith-1.5-35B-A3B-REAP-50-NVFP4A16) | 12.47 GiB | 84.15% [77.8, 88.9] | 89.15% [85.6, 91.9] | 22/50 = 44.0% | vLLM, two gaps patched (now upstream) |
| [ZAYA1-8B W4A4 uniform](https://huggingface.co/Ttimms/zaya1-8b-nvfp4-w4a4-uniform) | 6.02 GB | 72.6% [65.3, 78.8] &dagger; | -- | -- | pinned vLLM 0.20.2 + local W4A4-MoE patch; `enforce_eager` |

KAT and Ornith coding scores are greedy, instruct framing, on the released checkpoint. SWE-bench
Verified is `mini-swe-agent`'s bash-only scaffold at a 49K-token context ceiling on a fixed
50-instance slice -- the ceiling is what the 16 GB KV budget supports, so those runs are
context-limited and the resolved-of-completed rate (KAT ~81%, Ornith ~81%) is the better read
of patch quality.

&dagger; ZAYA1's figure is plain **HumanEval** (not HumanEval+), and it is a reasoning model
evaluated with budget forcing at temperature 0.6 rather than greedy -- see its
[write-up](https://github.com/t-timms/zaya1-nvfp4-w4a4) for the protocol and the full
statistics. It is in this table as a footprint reference, not a like-for-like comparison with
the two coders, and its serving path currently needs a pinned older stack (section 5).

What is distinct is narrow, and that is the whole claim: a **vLLM-servable build genuinely
usable in 16 GB**, with published SWE-bench Verified / HumanEval+ / MBPP+ numbers and their
confidence intervals. REAP-plus-NVFP4 on `qwen3_5_moe` already existed before this; what did
not exist was one you could run on a consumer card with the evaluation attached.

## 9. What I would tell someone starting this

- **Survey before the first action.** Search specifically for "is X broken on Y", not "does X
  exist". The SM120 NVFP4 MoE kernel situation is documented across several upstream issues; I
  found them after a night of debugging instead of before.
- **A discovered hazard is a search pattern.** When you learn that offloaded parameters
  silently discard writes, grep every other call site that writes to those parameters before
  fixing the one that bit you. The one that bit you failed loudly; the others will not.
- **Validate at the smallest scale first.** Every expensive failure in this project would have
  been caught by a 2-batch dry run of the exact invocation. None needed an hours-long run to
  surface.
- **Read the artifact, not the log.** "2000/2000 calibrated, 0 repaired" was true of the
  in-process state and false of what reached disk. If something must persist, re-open the file
  and check.

## 10. Follow-up (Nov 2026): when pruning stops being worth it

I ran the same pipeline on a third `qwen3_5_moe` base,
[Sage-A1.5](https://huggingface.co/InternScience/Sage-A1.5) (35B-A3B), this time targeting
llama.cpp GGUF instead of vLLM. I didn't release it. Here is why.

Two things changed. llama.cpp gained an **MoE expert cache** (`--moe-cache-mib`). Expert weights stay in host RAM and a fixed-size VRAM cache holds the
experts in use, so an unpruned Q4_K_M (21 GB) now runs on a 16 GB card. Separately, a public mixed-precision quant
([APEX-I-NanoPlus](https://hf.co/IsValorum/Sage-A1.5-APEX-I-NanoPlus-V2.1-GGUF), 13 GB) fits
entirely in VRAM without removing any experts.

Measured on one RTX 5070 Ti (Ryzen 5 7600, WSL2 VM capped at 64 GB RAM), llama.cpp build 1536 (`e01c9f18`), with the
same script for every file in one session. Setup: one slot, ctx 32768, `-fa on -t 6`, the model
card's sampling, MTP off. Decode is the median of 3 requests of 768 tokens. Prompt read is the median of
2 requests with a 14,812-token prompt. KLD is measured against the unpruned bf16 logits at ctx 512,
on held-out Mixture-of-Thoughts (in-domain for the prune calibration) and on wikitext-2 test.

| file | args | decode tok/s | prompt read tok/s | VRAM | KLD in-domain / wikitext |
|---|---|---:|---:|---:|---|
| unpruned [official Q4_K_M](https://hf.co/InternScience/Sage-A1.5-Q4_K_M-GGUF), 21.17 GB | `-ngl 99 -cmoe --moe-cache-mib 11968` | 93.6 | 652 | 15.6 GB | **0.0166 / 0.0351** |
| unpruned Q4_K_M | `-ngl 99 -cmoe --moe-cache-mib 9280` | 83.8 | 531 | 13.0 GB | same |
| unpruned Q4_K_M | `-fit on -fitt 1024` (no cache) | 67.6 | 1054 | 15.0 GB | same |
| APEX-I-NanoPlus, 13.03 GB | `-fit on -fitt 1024` | 152.1 | 5152 | 13.7 GB | 0.0366 / 0.0872 |
| my REAP-37 i1-IQ4_XS, 12.28 GB | `-fit on -fitt 1024` | 155.7 | 6432 | 13.5 GB | 0.0331 / 0.4018 |

- **For quality, the unpruned model with the cache wins outright.** It decodes at 94 tok/s, but
  needs ~20 GB of host RAM and reads long prompts 8-10x slower than the 13 GB files.
  At 15.6 GB, the 11968 MiB setting won't fit next to a game or OBS; 9280 is the everyday setting.
- **At ~13 GB, APEX ties the prune on speed (+2 %).** The prune's KL divergence on general text
  (wikitext) is 4.6x worse. Removing 37 % of experts without recovery training damages text outside
  the calibration domain. Re-pruning with prose added to the calibration still left wikitext KLD
  about 4x APEX's. Adding prose to the imatrix text made no difference (0.3634 vs 0.3653).
- **The gate I now run before any MoE prune:** measure the unpruned Q4 with the expert cache
  on the target card first. Prune only if that is unusably slow, or if the base doesn't fit in
  VRAM + RAM at all.

**The Blackwell FP4 tensor cores in llama.cpp, measured.** llama.cpp has an NVFP4/MXFP4 kernel that uses
SM120's block-scaled FP4 MMA (`m16n8k64`). It only runs for batched prompt processing with 4-bit
activations. Token-by-token decode takes a different path. The same pruned model, quantized to plain
NVFP4 (12.89 GB), same script and `-fit on -fitt 1024`:

| setup | decode tok/s | prompt read tok/s | KLD in-domain / wikitext |
|---|---:|---:|---|
| NVFP4, native FP4 (4-bit activations) | 156.1 | **8007** | 0.0684 / 0.4633 |
| NVFP4, `GGML_CUDA_MMQ_PREC=q8` (8-bit activations) | 157.1 | 5512 | 0.0528 / 0.4357 |
| i1-IQ4_XS control | 155.2 | 6463 | 0.0331 / 0.4018 |

The FP4 cores are real: +45 % prompt reading on the same file. They add nothing to generation speed.
Plain (untuned) NVFP4 costs about 2x the in-domain KLD of an imatrix IQ4_XS. On this card, NVFP4-GGUF is a
prompt-speed option for long-context agent loops, not a quality or decode win. A tuned NVFP4
quantizer (imatrix + scale fitting) was not measured.

To reproduce: `llama-server` with the args in each row plus `-c 32768 -np 1 -fa on -t 6 --jinja --reasoning-format deepseek`;
KLD with `llama-perplexity --kl-divergence-base <bf16 logits> --kl-divergence -c 512`; the
NVFP4 file with `llama-quantize --tensor-type '^blk\.=nvfp4' <bf16.gguf> <out.gguf> Q4_K_M`.
Each speed row is one session's median, and KLD is one deterministic pass.

Code and full build trails: [github.com/t-timms](https://github.com/t-timms). The ZAYA1 work
has a standalone write-up ([`PAPER.md`](https://github.com/t-timms/zaya1-nvfp4-w4a4)) with the
statistics in full.
