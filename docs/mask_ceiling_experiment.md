# The mask-ceiling experiment (`2026_08_19_MaskCeiling`)

What is board-validity masking actually worth, and why does a masked descriptor never
beat its unmasked twin? This note is the full record of the experiment that answered it:
four 50-epoch arms on `iteration_4` track data, plus an evaluation-only re-scoring of
every resulting checkpoint under a common set of mask conditions.

The mechanism itself — why weighting an equivariant feature field by a scalar mask stays
equivariant, where the two entry points are, how the predictor is supervised — is
[steerable_masking.md](steerable_masking.md). This note is only about the measurement.

**Short version.** A perfect mask is worth ~2.7x, almost all of it at the input gate. The
predictor collects *none* of the gate's prize and about two thirds of the late weight's
much smaller one, so the shipped masked model ends up ~13% *worse* than never masking.
Training under a perfect mask does not recover the difference: it buys 1.35x over simply
handing the ordinarily-trained trunk a perfect mask at test time, it peaks in the first
two epochs and degrades for the next 48, and it leaves a model that is not a descriptor at
all — take its mask away and it scores 0.83 FPR95 where the unmasked control scores
0.035.


## The question

On `iteration_4` track data a masked run does not beat its unmasked twin. Best
`FPR95@all` from the earlier sweep:

| head | masked | unmasked |
| --- | --- | --- |
| `fft` | 0.036 | **0.035** |
| `bispectrum` | 0.034 | **0.030** |
| `relphase` | 0.037 | **0.036** |
| `efficient8` | 0.373 | **0.363** |
| `efficient4` | **0.379** | 0.388 |

Two very different readings fit that, and they imply opposite next steps:

1. **The mask buys nothing on this data.** Off-board background on a canonicalized track
   patch is a thin rim and often bland; there may simply be no information in suppressing
   it. → drop the mask path.
2. **The mask buys something the predictor cannot cash in.** → keep the path, fix the
   predictor.

A third reading was the one that prompted the experiment: **the trunk may be spending
capacity compensating for an imperfect mask** — learning to be robust to `m_pred`'s
errors instead of learning to describe the patch — in which case a trunk trained under the
*truth* would be a better descriptor, and the masked arm's failure would be an artefact of
training-time mask noise rather than of test-time mask noise.

`oracle_mask=True` exists to separate all three. It hands the model the true mask on
**every** view, target included. It is not deployable — the target's mask does not exist
at test time — it is the ceiling arm.


## The four arms

50 epochs, `iteration_4` s96 logpolar, `HardNetLogPolar` with `head: fft`,
`n_harmonics: 4`, batch 4096, AdamW 1e-4, SupCon, band-balanced sampler. The `cfg.yaml`
diffs between arms are single-variable (verified with `diff`):

| arm | config | masks with | what it is, in `mask_ablation.py` terms |
| --- | --- | --- | --- |
| `track_logpolar_fft_s96` | `..._logpolar_fft` | – | `none` |
| `track_logpolar_fftmask_s96` | `..._logpolar_fftmask` | GT on the PDF view, `m_pred` on the target | `native` |
| `track_logpolar_oracle_s96` | `..._logpolar_oracle` | GT on **every** view, **both** entry points | `both_gt` |
| `track_logpolar_oracle_gate_s96` | `+ cascade: true` | GT on every view, **input gate** only | `gate_gt` |

Note the third row. With `oracle_mask` the non-cascade forward *also* runs
`input_norm(patches, mask=mask)`, not just the late weight — so `logpolar_oracle` masks at
both points. The launcher's old "(late weight)" label was wrong and has been corrected.

Both oracle arms run with `mask_loss_weight: 0`. `m_pred` is still emitted and still
scored, so the predictor gets its diagnostic curve, but the BCE does not pull on the
shared trunk — otherwise the arm would differ from its control by two things at once.
`process_batch_blobs` also stops withholding the mask at validation for an oracle model:
an oracle model validated without its mask is a different model.

Submitted as:

```sh
./launch_track_matrix.sh \
  --track /raid/data/hsa/datasets/tracks/iteration_4/logpolar/96/iteration_4_logpolar_s96.tracks \
  -n maskceiling \
  logpolar_fft logpolar_fftmask logpolar_oracle logpolar_oracle_gate
```


## What the runs did

`FPR95@all` on the validation split, per epoch (abridged):

| epoch | `fft` | `fftmask` | `oracle` | `oracle_gate` |
|---|---|---|---|---|
| 0 | 0.0656 | 0.0684 | 0.0298 | **0.0128** |
| 1 | 0.0632 | 0.0532 | 0.0249 | 0.0129 |
| 2 | 0.0548 | 0.0517 | **0.0247** | 0.0129 |
| 3 | 0.0439 | 0.0573 | 0.0265 | 0.0135 |
| 5 | 0.0423 | 0.0567 | 0.0352 | 0.0179 |
| 10 | 0.0501 | 0.0483 | 0.0441 | 0.0189 |
| 20 | 0.0564 | 0.0580 | 0.0580 | 0.0385 |
| 30 | 0.0516 | 0.0525 | 0.0733 | 0.0513 |
| 40 | 0.0417 | 0.0449 | 0.1242 | 0.0443 |
| 49 | 0.0483 | 0.0539 | 0.1299 | 0.0533 |

| arm | best | at epoch | epoch 49 | train SupCon @49 |
| --- | --- | --- | --- | --- |
| `fft` | **0.0349** | 32 | 0.0483 | 0.752 |
| `fftmask` | 0.0397 | 41 | 0.0539 | 0.754 |
| `oracle` | 0.0247 | 2 | 0.1299 | 0.598 |
| `oracle_gate` | **0.0128** | 0 | 0.0533 | 0.626 |

Two things are already visible without any further analysis:

* **The oracle arms peak in their first two epochs** and degrade monotonically for the
  remaining 48. The controls do not — they improve to a plateau and wobble around it.
* **They degrade while their training loss keeps falling**, and falls further than the
  controls' (0.598/0.626 vs 0.752). The perfect mask makes the training objective easier
  to satisfy and what it buys does not transfer. That is the shape of a shortcut.

The mask BCE (validation, weight 0, diagnostic only) behaves as designed: `fftmask`'s
predictor learns (0.361 → 0.191), the oracle arms' predictors do not (0.649 → 0.733 and
0.877 → 0.842), because nothing trains them and nothing consumes them.


## Re-scoring: putting every arm on every condition

A training log only ever reports each arm under *its own* mask condition, which is exactly
the comparison that cannot answer the question. `src/mask_ablation.py` pushes one fixed
set of validation batches through a checkpoint with both mask entry points made explicit
and driven from a chosen source:

```
gate    what multiplies the PATCH, inside input_norm   (the `cascade` position)
weight  what multiplies the FEATURE FIELD before the head   (the late position)
```

each fed from `{nothing, the true mask, the model's own m_pred, another patch's true
mask}`. `none` — a literal 1 at both — is the condition a model's own `forward` cannot
express: called with `mask=None` it falls back to the *prediction*, not to no masking.

Because it bypasses `forward` and applies the gate/weight itself, it scores an oracle
model and a plain one identically under identical conditions.

Run on the same 5-board split (`08d 5b6 cf2 f75 f80`), the same seeded eval sampler
(`get_eval_batch_sampler(1024, confuser_fraction=0.5)`), the full 80 batches:

```sh
pixi run --manifest-path deps/BlobBoards.jl/pixi.toml python src/mask_ablation.py \
  --run .../track_logpolar_fft_s96 --run .../track_logpolar_fftmask_s96 \
  --run .../track_logpolar_oracle_s96 --run .../track_logpolar_oracle_gate_s96 \
  --tracks .../val_iter4_s96.tracks --checkpoint best --occlusion 0
```

**Footing check.** `gate_gt` on `oracle_gate` *is* what the training loop measured for
that arm, so the two must agree — 0.0129 here vs 0.0128 in the log at `best`, 0.0537 vs
0.0533 at `latest`. They do, to the third digit. The harness and `FPR95@all` are one
scale, so every number below can be compared with every training curve above.


## Results

`best.pth` (each arm's own early-stopped checkpoint):

| condition | `fft` | `fftmask` | `oracle` | `oracle_gate` |
| --- | --- | --- | --- | --- |
| `none` — no mask anywhere | 0.0353 | 0.0546 | **0.8333** | **0.5343** |
| `native` — GT on PDF, `m_pred` on target (what ships) | – | 0.0399 | – | – |
| `native_gt` — as shipped but the target's mask is perfect | 0.0292 | 0.0321 | 0.2406 | 0.1004 |
| `late_gt` | 0.0293 | 0.0321 | 0.2435 | 0.1017 |
| `late_pred` | – | 0.0399 | 0.8397 | 0.4046 |
| `gate_gt` | 0.0311 | 0.0174 | 0.0190 | **0.0129** |
| `gate_pred` | – | 0.0665 | 0.7487 | 0.7304 |
| `both_gt` | 0.0288 | 0.0318 | 0.0250 | 0.0189 |
| `late_shuffled` (control) | 0.0580 | 0.0689 | 0.7394 | 0.5318 |
| `gate_shuffled` (control) | 0.0751 | 0.4995 | 0.8886 | 0.5810 |

`latest.pth` (epoch 49):

| condition | `fft` | `fftmask` | `oracle` | `oracle_gate` |
| --- | --- | --- | --- | --- |
| `none` | 0.0488 | 0.0676 | 0.7550 | 0.9016 |
| `native` | – | 0.0543 | – | – |
| `late_gt` | 0.0414 | 0.0474 | 0.6088 | 0.2051 |
| `gate_gt` | 0.0439 | **0.0198** | 0.0869 | 0.0537 |
| `gate_pred` | – | 0.0898 | 0.7856 | 0.7018 |
| `both_gt` | 0.0422 | 0.0455 | 0.1309 | 0.0425 |

Predictor diagnostics (`fftmask`, held-out, target views only): `m_pred` mean 0.792
against a true 0.777, correlation 0.830, IoU@0.5 0.888, and the late weight moves the
descriptor by `|d(gt) − d(1)|/|d|` = 0.38, so the path is far from inert.

For reference, **chance on this metric is 0.95**: if positives and negatives are
indistinguishable, the threshold that recalls 95% of positives admits 95% of negatives.

### 1. A model trained under a perfect mask is not a descriptor

The `none` row is the finding. Take the mask away from either oracle arm and it scores
0.83 resp. 0.53 against 0.0353 for the unmasked control. It never learned to describe a
patch; it learned a function of *(patch, mask)*, and it spent its capacity on the half
that will not exist at test time.

One caveat on reading `none` for `fftmask`: that model trains with a gated PDF input, so
an ungated pass is off its manifold and 0.0546 overstates how much it *lost* by masking.
It does not affect the oracle arms' reading — there the gap to the control is 15–24x, not
a manifold effect — nor any `*_gt` row, which is what the ceiling arithmetic uses.

`gate_pred` says the same thing from the deployment side. Hand the oracle-gate model the
real predictor's output — correlation 0.83, IoU@0.5 0.89, which is not a bad mask — and it
collapses to 0.73. The gate has essentially no error tolerance for a model that was
trained never to encounter any.

### 2. The ceiling is real, and small

* A perfect input gate is worth **2.7x** over the best deployable model
  (0.0129 vs `fft`'s 0.0353), and 3.1x over the masked arm's own operating point (0.0399).
* Training under the oracle buys only **1.35x** over not doing it: give the ordinarily
  trained `fftmask` trunk a perfect gate *at test time* and it reaches 0.0174.
* And that 1.35x is the unstable half. `oracle_gate` holds 0.0129 for one epoch and ends
  at 0.0537; `fftmask`'s `gate_gt` barely moves — 0.0174 at `best`, 0.0198 at `latest`.

### 3. Which answers the question the arm was built for

**The trunk is not compensating for its imperfect mask in any way that costs measurable
accuracy.** There is no reservoir of performance being spent on robustness to `m_pred`'s
errors: the same trunk, handed a perfect mask it was never trained on, reaches 0.0174
against the oracle-trained 0.0129. The entire gap between the shipped 0.0399 and the 0.0174 ceiling is
the **predictor at test time**, not the trunk's training-time diet. Training with the
truth does not close it — it trades a small ceiling gain for total mask dependence.

### 4. The gate is the position that matters, and only for a trunk that saw one

`gate_gt` beats `late_gt` by 1.8x on `fftmask` (0.0174 vs 0.0321) and by 7.9x on
`oracle_gate` (0.0129 vs 0.1017) — the receptive-field argument in
[`cascade`](steerable_masking.md#cascade-the-gate-in-the-loop),
reproduced on trained checkpoints. But on the mask-free `fft` trunk the gate is the *worse*
position (0.0311 vs 0.0293): a trunk that has never seen a gated input gains almost nothing
from one, because gated inputs are off its manifold.

The mirror image is `gate_shuffled`. On `fft` a wrong gate costs 2.1x; on `fftmask` it
costs 9.2x (0.4995). The more a trunk exploits the gate, the less mask error it tolerates.


## Why `FPR95@all` and the per-band curves disagree

At epoch 49 `oracle_gate` beats `fft` in **all nine** obliquity bands while losing on
`FPR95@all`. That has to be explained before any of the above can be trusted.

| band | `fft` best / ep49 | `fftmask` best / ep49 | `oracle` best / ep49 | `oracle_gate` best / ep49 |
| --- | --- | --- | --- | --- |
| 00–10° | 0.0001 / 0.0002 | 0.0001 / 0.0002 | 0.0000 / 0.0001 | 0.0003 / 0.0000 |
| 10–20° | 0.0007 / 0.0008 | 0.0006 / 0.0008 | 0.0006 / 0.0022 | 0.0017 / 0.0009 |
| 20–30° | 0.0077 / 0.0091 | 0.0119 / 0.0130 | 0.0045 / 0.0049 | 0.0071 / 0.0037 |
| 30–40° | 0.0171 / 0.0164 | 0.0171 / 0.0282 | 0.0121 / 0.0106 | 0.0140 / 0.0074 |
| 40–50° | 0.0283 / 0.0269 | 0.0249 / 0.0289 | 0.0132 / 0.0086 | 0.0257 / **0.0060** |
| 50–60° | 0.0053 / 0.0057 | 0.0053 / 0.0076 | 0.0081 / 0.0039 | 0.0169 / 0.0030 |
| 60–70° | 0.0031 / 0.0035 | 0.0030 / 0.0053 | 0.0058 / 0.0036 | 0.0139 / 0.0028 |
| 70–80° | 0.0152 / 0.0195 | 0.0174 / 0.0188 | 0.0246 / 0.0100 | 0.0352 / 0.0071 |
| 80–90° | 0.0826 / 0.0742 | 0.1067 / 0.1048 | 0.0733 / 0.0397 | 0.1185 / **0.0185** |

### Hypothesis A: cross-obliquity matching — dead

The bands only ever compare observations of similar obliquity; `all` also has to match a
5° view against an 80° one. A scratch script built both populations by hand — one disjoint
observation pair per track, 1024-patch batches, the same confuser pool and seed, `within` =
both views in the same 10° band, `cross` = one below 20° and one above 50°, both
populations truncated to the same number of pairs — and scored `latest` on each:

| model | gate | within-band | across-band |
| --- | --- | --- | --- |
| `fft` | none | 0.0079 | 0.0085 |
| `fftmask` | none | 0.0097 | 0.0095 |
| `fftmask` | `gt` | 0.0264 | 0.0305 |
| `oracle` | `gt` | 0.1511 | 0.1722 |
| `oracle_gate` | `gt` | 0.0058 | 0.0057 |

No model pays a cross-angle penalty worth speaking of — `oracle_gate` pays none at all.
(Only the within/across comparison *within a row* is on a common footing here; this paired
population is a different and much easier problem than the eval sampler's, so the columns
are not comparable to `FPR95@all` and the ordering between rows does not carry over.)

### Hypothesis B: unposed observations — dead

The band datasets can only contain observations that carry a viewing angle; `all` contains
everything. If the arms differed on posed vs unposed data the split would show it. It
cannot: the same script found **0** track pairs among observations without a pose. Every
tracked observation in this split is posed.

### Hypothesis C: the positive tail — confirmed

The bands and the paired test above both compare **one pair per track**. `FPR95@all` packs
whole track groups, so a track with *n* observations contributes *n(n−1)/2* pairs and the
95%-recall threshold is set by the worst 5% of a much longer, much harder positive list.

`mask_ablation.py --tails` prints the distributions behind the metric (six eval batches,
`latest`):

| model | pos p50 | pos p95 | neg p50 | threshold@95% | FPR |
| --- | --- | --- | --- | --- | --- |
| `fft`, no mask | 0.552 | 1.111 | 1.401 | 1.108 | 0.0484 |
| `oracle_gate` + `gate_gt` | 0.431 | 0.972 | 1.416 | 0.970 | 0.0634 |
| `fftmask` + `gate_gt` | 0.280 | 0.623 | 1.345 | 0.623 | **0.0311** |

The oracle-trained model's positives are tighter than the control's at every quantile — it
really is the better matcher pair-for-pair, which is what the bands were reporting — but
its negatives do not move out proportionally, so the threshold lands deeper into their
tail. The ordinarily trained trunk handed the same perfect gate tightens positives *and*
keeps negatives out, which is why it wins the metric the model will actually be judged on.

Both splits of positives are dominated by image↔image pairs (97% of them); PDF↔image pairs
are a much easier problem and every arm is near-perfect on them (proxy-anchored FPR95
0.00004 for `oracle_gate` + `gate_gt`), so `FPR95@all` is effectively an image↔image
metric.


## Does the mask leak blob identity?

If the mask by itself identifies a blob, a mask-fed model can learn to encode the mask
instead of the patch — which would be a mechanism for everything above.
`mask_ablation.py --content-blind` tests it directly: every patch is replaced by **one
fixed random texture**, so the mask is the only thing that still varies between samples.
(A fixed *random field* rather than a constant: `input_norm` divides a flat patch by ~0
std, and a gate that fills the invalid region with the valid region's mean leaves a
constant patch constant, so nothing — not even the mask boundary — would reach the trunk.)

`best.pth`, occlusion 0:

| condition | `fft` | `fftmask` | `oracle` | `oracle_gate` |
| --- | --- | --- | --- | --- |
| `none` (all descriptors literally identical) | 0.3670 | 0.3649 | 0.3632 | 0.5280 |
| `late_gt` | 0.5128 | 0.4214 | 0.2425 | 0.3548 |
| `gate_gt` | 0.5814 | **0.1934** | **0.1874** | 0.3634 |
| `both_gt` | 0.5602 | 0.4045 | 0.2889 | 0.3491 |
| `gate_shuffled` (control) | 0.9483 | 0.9480 | 0.9493 | 0.9481 |

**The mask does carry blob identity.** With zero image content, the true mask alone gets a
mask-trained trunk to 0.19 where a shuffled mask gets 0.95, i.e. chance. The mask-free
`fft` trunk leaks nothing — its `gate_gt` (0.58) is *worse* than no gate, as expected for a
trunk that cannot read a gate at all.

Read this as present and non-trivial rather than as *the* mechanism, for two reasons:
0.19 is a long way from the 0.017 the same trunk reaches with content, and the `none` row
is not a clean floor — with identical inputs every descriptor is identical, so its value
is a tie-breaking artefact of the metric (which is also why `oracle_gate`'s differs).


## Verdict

* **Do not train with ground-truth masks.** The resulting weights are not a descriptor
  (`none` 0.83 / 0.53), the arm overfits from epoch 2 onward, and the ceiling gain over
  simply masking the ordinarily-trained trunk at test time is 1.35x.
* **The masking mechanism is sound and the ceiling is real** — a perfect input gate is
  worth 2.7x over the best deployable model, and ~10x on occluded patches (see
  [steerable_masking.md](steerable_masking.md#occlusion-the-test-background-masking-does-not-perform)).
* **The binding constraint is the predictor at test time.** At the input gate it collects
  *nothing* — `gate_pred` 0.0665, worse than the 0.0546 of no mask at all — because the
  gate tolerates almost no mask error. At the late weight it collects about two thirds of
  a much smaller prize (0.0546 → 0.0399 against a `late_gt` ceiling of 0.0321).
* **As it ships today the mask path costs rather than pays**: 0.0399 against 0.0353 for
  never having masked, ~13% worse, consistent across every head in the earlier sweep.

So the useful next questions are all about the predictor, not the trunk: a target that is
not just board-coverage (the predictor is structurally blind to occlusion because its BCE
label is the board-in-frame mask), a higher-resolution or better-tapped mask head, or a
gate that degrades gracefully toward 1 as predictor confidence drops. Nothing here
suggests the trunk is the problem.


## Reproducing it

The `.tracks` file is 8 GB and lives on the DGX; the ablation needs only the five
validation boards. Sequence names are `<media>_<uid>` or `<uid>`, so the subset has to be
selected the way `track._select_sequences` does — on the trailing uid, not the group name:

```python
import re, sys, h5py
uid = lambda n: (re.fullmatch(r"(?:.*_)?([A-Fa-f0-9]+)", n) or [None, None])[1]
src, dst, boards = sys.argv[1], sys.argv[2], set(sys.argv[3:])
with h5py.File(src, "r") as f, h5py.File(dst, "w") as o:
    og = o.create_group("sequences")
    for n in [s for s in f["sequences"] if uid(s) in boards]:
        f.copy(f["sequences"][n], og, name=n)
```

Taking only the groups literally named `08d`, `5b6`, … yields 59 MB of **PDF renders
only** (`is_anchor: 1`, no observations); the correct subset is 962 MB and 125 sequences.

Then, in the pixi env:

```sh
# the matrix (table under "Results")
python src/mask_ablation.py --tracks val.tracks --checkpoint best --occlusion 0 \
  --run .../track_logpolar_fft_s96 --run .../track_logpolar_fftmask_s96 \
  --run .../track_logpolar_oracle_s96 --run .../track_logpolar_oracle_gate_s96

# the leakage control
python src/mask_ablation.py --tracks val.tracks --occlusion 0 --content-blind --run ...

# the distance tails
python src/mask_ablation.py --tracks val.tracks --checkpoint latest --occlusion \
  --tails --max-batches 6 --run ...
```

On a 12 GB card set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the harness
chunks the forward pass at 128 patches, which is exact (`input_norm` and the head are
per-patch), so FPR95 still ranks the whole 1024-patch batch.


## What this did not test

* **Only `HardNetLogPolar`.** `oracle_mask` is implemented for both steerable descriptors
  too, but `mask_ablation.py` hooks the log-polar family only — the steerable models apply
  their weight inside `self.net` and need their own hook. The earlier sweep's masked-vs-
  unmasked gap is the same shape there (`efficient8` 0.373 vs 0.363), so the conclusion
  plausibly transfers, but it is not measured.
* **Only one head** (`fft`, `n_harmonics: 4`) and one scale (96).
* **One seed per arm.** The `fft`/`fftmask` epoch-to-epoch wobble is ±0.01 on a best of
  0.035, which is the same order as the 0.0349-vs-0.0397 gap between them; the oracle-arm
  effects (0.0128, 0.83) are far outside it, but the control-vs-masked ordering itself
  rests on the earlier five-head sweep agreeing, not on this run alone.
* **No occlusion sweep on the oracle arms.** The occlusion results in
  [steerable_masking.md](steerable_masking.md#occlusion-the-test-background-masking-does-not-perform)
  are on the earlier `fftmask`/`fft` checkpoints.
