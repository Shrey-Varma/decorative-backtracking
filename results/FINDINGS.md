# Verbalized self-correction is dissociated from answer improvement in an R1 distill

**Model:** DeepSeek-R1-Distill-Llama-8B · **SAE:** andreuka18 JumpReLU, residual stream layer 19 · **Data:** MATH-500, 800 chains

## Thesis (positive framing)

In a reasoning distill that constantly narrates reconsideration ("wait…",
"hmm", "but wait"), the *verbalized* self-correction is **behaviorally and
causally dissociated** from the computation that actually improves the answer.
The model says it is reconsidering far more often, and in different places,
than its answer actually moves — and removing the layer-19 SAE feature that
fires on those verbalizations does not change where it lands.

We lead with the **localization/dissociation** result (a positive measurement)
and use the feature knockout as **corroborating causal evidence**, rather than
resting the claim on a single-feature null.

## Three independent lines of evidence

### 1. Behavioral dissociation (primary)

- R1-distill verbalizes reconsideration **~19× per chain** (14,230 marker
  events across 751 chains that have any).
- Yet across 1,699 wrong→right answer transitions in the force-answer
  trajectories, only **32%** are aligned to a correction event. **Two-thirds of
  the answer improvement happens *between* the "wait" moments, not at them.**
- This is measured within-chain against the intended-answer trajectory, so it
  is not a position artifact (see Methods — the position confound is excluded
  by construction).

### 2. Correlational (feature selection)

- The correction-vocabulary features have answer-flip AUROC ≈ **0.46–0.54**
  (chance). The feature that fires on "wait" does **not** predict whether the
  intended answer is about to change.

### 3. Causal (knockout-and-continue)

Ablating the correction feature at the onset of the verbalized correction and
letting the model finish (K=8 seeds × 60 wrong→right events, error-preserving
ablation, all controls):

| split | Δ acc (natural − ablate) | p | verdict |
|---|---|---|---|
| pilot | −0.049 | 0.16 | decorative |
| **test (confirmatory)** | **−0.005** | **0.94** | **decorative** |

- REVERT rate <1%; every condition (ablate, placebo-feature, placebo-position,
  natural) lands NO_CHANGE ~70%. `ablate_correction` is statistically
  indistinguishable from both placebos → the small effects are generic
  disruption, not correction-specific.
- Decision rule **pre-registered** in config before the test split was
  unsealed; Δ is ~60× below the 0.30 faithfulness threshold.

### 4. Causal — text-level excision (no localization assumption)

The ablation results are causal nulls *about their carriers*; to remove the
"maybe you localized the wrong carrier" premise entirely, we intervene in
**text space**. On the same 60 events, all on one backend with shared seeds:

| arm | test acc | pooled acc |
|---|---|---|
| A · with-correction (continue after the correction span) | 0.716 | 0.765 |
| B · without-correction (resample from the onset) | 0.703 | 0.749 |
| C · neutral-filler (span → length-matched neutral text) | 0.659 | 0.744 |

- **A − B = +1.6%** (pooled): having verbalized the correction is no more
  accurate than resampling from its onset. TOST-equivalent to zero within
  ±10% (p = 0.005), **BF₀₁ = 4.3**.
- **A − C = +2.2%** (pooled; length-controlled): the real correction beats a
  same-length neutral filler by only ~2 points.
- Whatever internal carrier the correction uses, its downstream text-level
  consequence is ≤2% accuracy — decorative, with no localization assumption.
- Caveat: the filler ("…continuing with the same approach…") is imperfectly
  neutral (leans against re-opening), so A − C is an upper bound; A − B is the
  clean contrast and it is ~0.

### 5. Causal — sufficiency (insertion & carrier clamp-ON)

Necessity (§3–4) shows removing the correction doesn't hurt. Sufficiency asks
the mirror: at a point where the intended answer is **wrong**, does *adding* a
verbalized correction rescue it? Two populations, K=8 seeds, shared backend:

**Text insertion — splice a synthetic "wait, I made a mistake…" into a
wrong-track prefix** (three phrasings, so the result can't hinge on wording):

| arm | failed-chain acc | wrong-point acc |
|---|---|---|
| insert_natural (no splice) | **0.056** | 0.696 |
| insert_correction · assert-mistake | 0.056 | 0.692 |
| insert_correction · soft-recheck | 0.052 | 0.685 |
| insert_correction · minimal-marker | 0.058 | 0.694 |
| insert_filler (length-matched) | 0.054 | 0.667 |

- **Failed chains** (final answer wrong — the model never rescued itself, *no
  ceiling*): inserting a correction gives **5.6% rescue vs 5.6% for nothing**.
  All three phrasings TOST-equivalent to zero (BF₀₁ = 4.5–5.9). This is the
  cleanest sufficiency null available — where a rescue would be most visible,
  the verbalized correction does **exactly nothing**.
- **Wrong points** in self-correcting chains (~70% base rescue = ceiling):
  corrections hold at ~69% (no lift); the length-matched filler *lowers* it to
  67%, so extra tokens alone if anything hurt.
- No phrasing helped (one-sided "helps" p ≥ 0.68 everywhere), so it is not the
  specific wording, and asserting "an error exists" was no better than a bare
  "hmm, wait" — the *information* doesn't help either.

**Hooked carrier clamp-ON** (turn the SAE feature / direction ON at the wrong
point, dose ladder 1×/2×): every arm is statistically flat vs `natural_at_insert`
(0.717) — clamp 0.704/0.696, direction-add 0.721/0.719; all TOST-equivalent,
BF₀₁ = 3.1–4.6, no dose-response. **Honest caveat:** the clamp did *not*
reliably induce the verbalization (marker rate 0.47–0.53 vs 0.485 base — it
fails the pre-registered ≥2× manipulation check), so the hooked arm bounds only
the tested doses; the text insertion, where the verbalization is present by
construction, is the clean sufficiency test. That the feature selected on
"wait" does not *generate* "wait" when clamped on is itself consistent with a
correlate, not a driver.

**Sufficiency verdict: not sufficient** — decisively at the text level, on the
no-ceiling population, invariant to phrasing and to asserted content.

### 6. Internal state change without answer-relevance (surface-text objection)

Are the markers *just* surface text, or do they mark a real internal
transition — and if real, does that transition predict the fix? One forward
pass per chain, representational shift `1 − cos(mean resid before, mean resid
after)` across the correction span, vs **sentence-start** pseudo-spans
(position-matched, event-clear — subtracts "state changes at discourse
boundaries", not merely depth):

| layer | event shift | control shift | Δ | Cohen's d | p |
|---|---|---|---|---|---|
| 15 | 0.471 | 0.419 | +0.056 | 0.70 | <0.001 |
| 19 | 0.500 | 0.443 | +0.062 | 0.73 | <0.001 |
| 23 | 0.475 | 0.414 | +0.064 | 0.76 | <0.001 |

- **The internal shift is real** (d ≈ 0.7 over three layers) — corrections are
  **not** pure surface text; something genuinely changes in the residual stream.
- **But it does not predict the answer.** AUROC for "does the correction
  direction / shift magnitude at the onset predict a wrong→right flip?" is
  **0.44–0.49** at every layer — at or below chance. The state moves; the
  movement is orthogonal to whether the answer improves.

This closes both halves of the surface-vs-substance question at once: the
marker is internally substantive *and* answer-decoupled — exactly the profile
of a decorative signal, not a load-bearing one.

## Why the null is defensible (addressing the central objection)

A reviewer will note that a null for *one vocabulary-selected feature at one
layer* could mean "we mislocalized," not "self-correction is decorative." We
address this directly:

- **Not just the top feature (§5 robustness):** we ablated candidates #2–5
  individually and all top-5 jointly. On the test split every Δ is within noise
  of zero (cand #2–5: −0.012, +0.014, −0.010, +0.012; **joint: −0.001**) —
  none is load-bearing, and a mechanism distributed across the shortlist would
  have surfaced in the joint arm but did not.
- **Not just this SAE / not just one localization:** an SAE-free
  difference-in-means "correction direction" ablation reproduces the null with
  a completely different (vocabulary-independent) localization method.
- **Layer sensitivity:** the direction ablation swept layers {15, 19, 23};
  test-split Δ = **−0.087, −0.074, −0.057** — all *negative* (knocking out the
  correction direction if anything slightly *raises* accuracy and *reduces*
  disruption: NO_CHANGE 0.78 vs 0.70 natural), the opposite of a load-bearing
  correction, at every layer. (An SAE layer-sweep is impossible — every
  off-the-shelf SAE for this model is at L19; we do not train new ones per the
  spec.)
- **Text-level closer (no localization at all):** excising the verbalized
  correction from context (arm B) costs ≤2% accuracy vs keeping it (arm A) —
  so the conclusion does not even depend on having found the right internal
  carrier.
- **What the null does NOT claim:** it does not claim no correction mechanism
  exists anywhere. It claims the verbalized-correction signal — localized by
  **two internal methods across three layers and six SAE features, and tested
  directly in text space** — is not load-bearing for the final answer on this
  benchmark.

## Causal, not correlational (summary)

The claim is established by *interventions*, not observation, on **both**
sides of the faithfulness definition:

- **Necessity** — removing the correction doesn't change the answer: at the
  **feature** (SAE knockout, null), **subspace** (direction ablation, null —
  slightly anti-faithful), and **text** (excision, assumption-free null) level.
- **Sufficiency** — adding the correction doesn't rescue a wrong answer: at
  the **text** level (insertion, no-ceiling failed chains, 5.6%→5.6%, three
  phrasings) and the **carrier** level (clamp-ON, flat; manipulation-check
  caveated).

Triangulated with (a) chance-level predictive AUROC, (b) chance-level flip
localization, and (c) a real-but-answer-decoupled internal state change
(d ≈ 0.7, prediction AUROC ≈ 0.46). The verbalized self-correction is
**neither necessary nor sufficient** for the final answer, is not where the
answer moves, and its internal correlate does not predict the move. The one
acknowledged limitation is generality (single model, single benchmark, SAE at
L19 only — mitigated by the SAE-free direction sweep at 15/19/23), not a hole
in the causal argument.

## Statistical inference (null affirmed, not merely unrejected)

All computed over per-chain paired deltas; confirmatory numbers from the
held-out test split (`analysis/stat_tests.json`):

- **Equivalence (TOST):** the primary ablation is formally equivalent to zero
  within ±0.10 (p = 0.0095); 9/13 arms pass two-sided equivalence, and **all
  13 arms** satisfy the one-sided bound Δ < +0.10 — the direction faithfulness
  predicts. On the pooled split, 10/13 pass two-sided TOST.
- **Bayes factors (JZS):** BF₀₁ ≈ 3.5–4.7 per SAE arm — moderate evidence for
  the exact null against *any* effect (and the pre-registered faithful effect
  was 0.30, far outside every CI). The text-level with-vs-without contrast is
  likewise equivalent to zero (BF₀₁ = 4.3).
- **Family-wise test:** across all 10 SAE-feature arms jointly, max-|Δ|
  permutation p = **0.996** (test) / 0.82 (pooled) — the largest observed
  effect anywhere in the SAE family is exactly what chance predicts.
- **Power:** minimum detectable effect at 80% power = 0.15 (test split) /
  0.10 (pooled) — 2–3× smaller than the pre-registered faithfulness threshold
  (0.30). The null is well-powered, not underpowered.
- **Localization is at chance:** observed event-aligned flip fraction 32.1%
  vs 31.4% expected under random flip placement (permutation p = 0.75).
  Verbalized corrections predict where the answer moves **no better than
  chance**.
- **Robustness of the null:** invariant to DISRUPTED handling (Δ with
  truncated rows excluded: **−0.0002**; as-correct bound: −0.006); disruption
  symmetric across conditions (p = 0.47); no per-event heterogeneity — no
  hidden faithful subpopulation (overdispersion p = 0.22); problem-level
  clustered bootstrap CI [−0.072, +0.017] still spans zero.

## Exploratory finding: removing the correction *direction* helps

The SAE-free direction ablation shows a significant effect in the
**anti-faithful** direction: Δ = −0.06…−0.09 across layers 15/19/23
(direction-family max-|Δ| permutation p = 0.0011 pooled, 0.051 test-only;
NO_CHANGE 0.75–0.78 vs 0.70 natural; DISRUPTED 0.20–0.24 vs 0.29). Removing
the reconsideration direction makes continuations *more* likely to conclude
correctly within budget — consistent with suppressed re-opening of reasoning
("overthinking"), and the opposite of what a load-bearing correction predicts.
Framed as exploratory (these arms were added post-registration), and partially
mediated by fewer token-cap truncations; a follow-up could disentangle
accuracy-per-token from completion-rate effects.

## Methods guardrails (why this isn't the predecessor's confound)

- Every causal contrast is **within-chain, same-truncation-point, same-K-seeds**
  — ablate vs natural differ only by the intervention, so chain position cannot
  explain Δ.
- **Placebo-position control** ablates the same feature at a matched non-event
  prefix; **placebo-feature control** ablates a corpus-matched non-correction
  feature. Both are flat.
- **Error-preserving ablation**: only the target feature's contribution is
  removed; reconstruction residual is preserved, so imperfect SAE
  reconstruction cannot corrupt the intervention.
- **DISRUPTED bookkeeping**: outcomes that land on neither the pre-correction
  nor the correct answer are tracked separately (never counted as
  faithfulness); ~28%, symmetric across conditions (token-cap truncation).

## Event-detection validation (§11 — closed)

A 50-event random sample of detected correction events was labeled by an LLM
judge (Claude Opus 4.8, single-rater, rubric in
`analysis/event_validation_summary.json`): **precision = 0.88** (conservative
bound 0.76 with every borderline flipped; 0.92 upper). Effect on the null:
~12% contamination by non-correction markers can only *attenuate* a real
per-correction effect by ~12% relative — irrelevant against a measured
Δ ≈ 0 vs a 0.30 pre-registered threshold. Recall of the lexical detector
against marker-free reconsideration is not measured (stated limitation).

## Open items

- **Single-layer/-model bound remains** for anything a direction-ablation at
  15/19/23 doesn't cover; single benchmark (MATH-500), single model — stated
  limitations, not runnable within budget.
- **Hooked sufficiency manipulation check:** clamping the SAE feature ON did
  not reliably induce the verbalization, so the hooked-arm sufficiency null is
  dose-bounded; the text-insertion null (verbalization present by construction)
  carries the claim. A cleaner hooked test would search for a clamp
  configuration that passes the induction check first.
