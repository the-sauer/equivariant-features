# Per-band FPR95: where the obliquity falloff actually comes from

Every `track_descriptor_*` run validates against `conf/track_angle_validation.yaml`, which
builds one validation split per 10° band of viewing obliquity and reports `FPR95@degXX_YY`
next to `FPR95@all`. This note collects those numbers across **50 runs** under `~/dgx/logs`
(9 different `.tracks` files) and asks one question:

> Is the descriptor's degradation on oblique views a **data** effect — too few oblique
> training pairs — or an **intrinsic** one the sampler cannot fix?

Short answer: intrinsic, on the evidence available. The falloff does not follow the
training population, and the one clean A/B of `balance_view_angles` made *every* band
worse, oblique bands included. A second, unrelated finding fell out of the same scan:
`BlobDescriptorEfficient`'s weakness is **not** specific to perspective — it is a uniform
gap that is *largest* at fronto-parallel, and it is dominated by patch scale.

Numbers below are the per-band FPR95 at each run's best epoch (best by `FPR95@all`, the
criterion `best.pth` uses). Reproduce with the scan scripts described at the end.

## How to read a per-band number

Each band split is a separate `BlobTrackData` with `view_angle_range: [lo, hi)`, capped at
`max_keypoints: 6000` per kind, and evaluated with the eval batch sampler (whole track
groups filling half the batch, distinct confusers the other half). Three consequences:

- **Bands are not equally composed.** Observations per track fall off with obliquity — on
  `iteration_4_cartesian_s96` the validation split runs 3.06 obs/track in `[0,10)` down to
  1.34 in `[80,90)`. A small group is mostly a *pdf↔observation* pair, which is the
  easiest positive there is; a large group adds hard observation↔observation positives.
  The composition therefore makes the **oblique bands look easier than they are**, so the
  measured falloff is a lower bound, not an artifact.
- **The tail is thin.** `[80,90)` holds 167 observations / 125 tracks on the cartesian
  s96 file and 76 / 76 on the log-polar one. Treat that band as indicative only.
- **`FPR95@all` is not comparable with the bands.** It has no `max_keypoints` cap and its
  track groups are un-truncated, so its positives are much harder. Compare bands with
  bands, `all` with `all`.

## 1. The falloff does not track the training population

Rank correlation of a run's per-band FPR95 against (a) the band index, (b) the band's
**training** population, over the 39 runs with ≥20 evaluations:

| | correlation with per-band FPR95 |
|---|---|
| band index (obliquity) | **+0.15 … +0.88**, positive in all 39 runs |
| training observations in that band | −0.35 … +0.63, **no consistent sign** |

The training distribution peaks at 20–40° and falls off by ~100x into the tail
(`iteration_4_cartesian_s96`: 15 413 / 43 081 / 48 641 / 48 673 / 41 813 / 31 402 /
14 044 / 3 859 / **462** observations per band). If scarcity drove the falloff, the worst
bands would be the emptiest ones. They are not — FPR95 is already 2-3x its `[0,10)` value
at 20–40°, which is where the data is *most* abundant.

## 2. Balancing the bands made every band worse

`2026_07_28_RealSecondIterationAngles` and `2026_08_08_ReaThirdIterationAsymmetricLoss`
are the same four models on the same file (`iteration_2_logpolar_s96.tracks`), 50 epochs
each, with configs that differ in **exactly one key** (`diff` of the two `cfg.yaml`s is
`balance_view_angles: true` plus the two band settings it needs):

| run | balance | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bispectrum | off | .0007 | .0014 | .0020 | .0038 | .0060 | .0035 | .0015 | .0021 | .0054 | .0040 |
| bispectrum | **on** | .0007 | .0021 | .0044 | .0064 | .0135 | .0067 | .0027 | .0040 | .0078 | .0201 |
| bispectrum+mask | off | .0007 | .0013 | .0020 | .0031 | .0049 | .0032 | .0014 | .0021 | .0043 | .0036 |
| bispectrum+mask | **on** | .0007 | .0019 | .0056 | .0089 | .0178 | .0081 | .0029 | .0033 | .0050 | .0245 |
| relphase | off | .0008 | .0017 | .0025 | .0038 | .0061 | .0040 | .0017 | .0026 | .0051 | .0055 |
| relphase | **on** | .0007 | .0027 | .0056 | .0082 | .0142 | .0082 | .0039 | .0047 | .0060 | .0331 |
| relphase+mask | off | .0008 | .0015 | .0029 | .0048 | .0064 | .0048 | .0019 | .0021 | .0042 | .0042 |
| relphase+mask | **on** | .0008 | .0034 | .0068 | .0083 | .0184 | .0088 | .0046 | .0050 | .0065 | .0387 |

Balanced training is **2-3.5x worse in every band above 10°** and 5-9x worse on `all`, in
all four architectures. It is not undertrained: the balanced epoch is *longer* (median
band pool 61 310, quota 113 ⇒ 543 batches/epoch, against ~247 for the group packer at
`batch_size: 4096`), so those runs took more optimizer steps, not fewer.

**Two candidate causes, and they are separable.** `_pack_band_balanced` changes two things
at once relative to `_pack_balanced`:

1. the **angle distribution** of the positives (the intended change), and
2. the **positive structure** — disjoint *pairs* (2 patches per track) instead of whole
   track groups (many views per track). For SupCon that is a materially different
   objective: far more classes per batch, two views each, no multi-view positives.

Cause 2 is the more likely culprit given that the regression is uniform across bands
(a reweighting should have helped *somewhere*). It is testable with the knobs that already
exist: run the balanced path with `view_angle_band_edges: [0, 90]` — one single band, so
the pair structure applies with **no** angle reweighting at all. If that run reproduces the
regression, the sampler's pair packing is the problem and the obliquity weighting is
innocent; if it lands on the unbalanced curve, the reweighting itself hurts.

Until that is run, `view_angle_band_bias > 0` pushes harder on a mechanism that has so far
only cost accuracy — worth a single probe run, not a sweep.

## 3. `BlobDescriptorEfficient` is not specifically bad at perspective

Best matched pair available, both balanced, `iteration_4` s96 (different files: the
efficient family is only ever trained on cartesian patches, the log-polar family only on
log-polar ones):

| | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | all |
|---|---|---|---|---|---|---|---|---|---|---|
| `BlobDescriptorEfficient` n_rot=8, s96 | .0585 | .1111 | .1556 | .1647 | .1594 | .1705 | .1621 | .1529 | .1364 | .3635 |
| `HardNetLogPolar` fft, s96 | .0001 | .0007 | .0077 | .0171 | .0283 | .0053 | .0031 | .0152 | .0826 | .0349 |
| ratio | **413x** | 171x | 20x | 10x | 6x | 32x | 53x | 10x | **2x** | 10x |

The gap is **largest at fronto-parallel and smallest at 80-90°**. Relative degradation
across the bands is *milder* for the efficient model (0-10 → 40-50: 2.7x) than for the
log-polar one (283x). Whatever is wrong with the efficient descriptor, "it does not handle
perspective" is not the shape of it: it is a uniform deficit that oblique views actually
narrow.

**Patch scale explains most of that deficit.** The `2026_08_20/21_Scales` sweep runs the
same `BlobDescriptorEfficient` at six scales (each on its own `iteration_4` file):

| scale | 0-10 | 40-50 | 80-90 | all | best epoch |
|---|---|---|---|---|---|
| 8 | .0117 | .0952 | .0188 | .0902 | 12 |
| **16** | **.0019** | **.0468** | **.0913** | **.0525** | 10 |
| 32 | .0074 | .1214 | .1909 | .0898 | 6 |
| 64 | .0187 | .1221 | .2571 | .0946 | 2 |
| 96 | .1285 | .3057 | .3272 | .2529 | 5 |
| 128 | .0465 | .1956 | .2051 | .1220 | 2 |

(The s8/s16 runs stopped after 18-19 evaluations rather than 50, and the s96 row is
`2026_08_20_Scales`, a different run from the s96 one in the table above — that one has
better per-band numbers and a worse `FPR95@all`, and was used above because it is the
efficient model's *best* per-band showing, i.e. the comparison is set up in its favour.)

At s16 its whole profile (.0019 / .0468 / .0913) sits within 2-3x of `HardNetLogPolar`'s
at s96 (.0001 / .0283 / .0826) — the same curve, not a different regime. The 400x gap in
the table above is a **scale** effect, not an architecture-vs-perspective effect.

Also visible: the s96/s64/s128 runs peak at **epoch 2-6 of 50** and degrade afterwards,
which is a training-stability problem worth its own look.

## What this cannot answer

Architecture, patch type and data file are **fully confounded** in every existing run:
`BlobDescriptorEfficient` only ever sees cartesian patches, `HardNet*LogPolar` only
log-polar ones, and each patch type has its own `.tracks` file with its own boards and
tracks. No crossover run exists. Any statement of the form "architecture A handles
perspective better than B" therefore rests on an untested assumption, and the honest
version is "pipeline A (representation + model + scale) scores better in band X".

To settle it, in increasing cost:

1. **Single-band balanced run** (`view_angle_band_edges: [0, 90]`) — separates the
   sampler's pair structure from its angle reweighting. One run, answers §2.
2. **Scale-matched cross-family run** — `BlobDescriptorEfficient` at s16 against
   `HardNetLogPolar` at s16, so the s96 comparison stops carrying the scale effect.
3. **Composition-matched band evaluation** — re-evaluate the saved checkpoints per band
   with the track groups truncated to a fixed size (and `max_keypoints` set to the
   smallest band), so the bands stop differing in how many hard observation↔observation
   positives they contain.

  Setting view_angle_band_edges: [0, 90] makes it one band covering all the data. The balanced code path still runs — same
  pair-structured positives, same epoch-length formula — but with a single band there is nothing to reweight; it draws
  uniformly from the whole pool, exactly like the file's own distribution. Verified on the test fixture:

  BlobTrackData train sampler (view-angle balanced): 1/1 populated bands x ~6 in-band patches per batch ...
    band [0, 90) deg: 12 obs, 12 tracks, quota 6/batch, 1.00 draws/obs per epoch

  So three arms of one architecture at one scale:

  ┌────────────┬──────────────────────────────────────────────────────┬────────────────────┬──────────────────────┐
  │    arm     │                       override                       │     positives      │   angle weighting    │
  ├────────────┼──────────────────────────────────────────────────────┼────────────────────┼──────────────────────┤
  │ A baseline │ training.dataset.params.balance_view_angles=false    │ whole track groups │ file's own           │
  ├────────────┼──────────────────────────────────────────────────────┼────────────────────┼──────────────────────┤
  │ B probe    │ training.dataset.params.view_angle_band_edges=[0,90] │ disjoint pairs     │ file's own           │
  ├────────────┼──────────────────────────────────────────────────────┼────────────────────┼──────────────────────┤
  │ C current  │ (defaults)                                           │ disjoint pairs     │ uniform over 9 bands │
  └────────────┴──────────────────────────────────────────────────────┴────────────────────┴──────────────────────┘

  Reading it: B lands on C ⇒ the pair packing is what cost the accuracy, the angle reweighting is innocent, and
  view_angle_band_bias is worth exploring once the packing is fixed. B lands on A ⇒ the reweighting itself is the damage,
  and a positive bias will make it worse — don't sweep it.

  Both override forms compose cleanly (checked against track_descriptor_logpolar). Via the launcher, on the current file:

  T=/raid/data/hsa/datasets/tracks/iteration_4/logpolar/96/iteration_4_logpolar_s96.tracks
  EXTRA_OVERRIDES="training.dataset.params.balance_view_angles=false" \
    ./launch_track_matrix.sh --track $T --scales 96 -n SamplerProbeGroups logpolar_fft
  EXTRA_OVERRIDES="training.dataset.params.view_angle_band_edges=[0,90]" \
    ./launch_track_matrix.sh --track $T --scales 96 -n SamplerProbeOneBand logpolar_fft
  ./launch_track_matrix.sh --track $T --scales 96 -n SamplerProbeBanded logpolar_fft


## Reproducing

The scan scripts live in the session scratchpad rather than the repo (they only read
`~/dgx/logs` and the `.tracks` files):

- `scan_runs.py` — walks the log tree, parses `cfg.yaml` plus both validation log formats
  (`finished epoch [n/N]` and the newer sub-epoch `validation at epoch x.xxx [n/N]`), and
  emits one JSON record per run with its full metric history.
- `band_counts.py` — per-band observation/track counts of each `.tracks` file's train and
  validation splits, straight from `view_angles` + `frame_id` + `homography_frame_ids`.
- `report.py` — the per-band tables above, grouped by the file a run trained on.
