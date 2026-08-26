# Necessity-test analysis

## Split: pilot

- Δ ablate_correction: **-0.0489** (CI (-0.1117710690045249, 0.009893701267056399))
- Δ control feature: -0.0375 (CI (-0.0826934523809524, 0.006673611111111072))
- Δ control position: -0.0050 (CI (-0.08856336805555552, 0.06514391447368414))
- faithfulness rate (net of control DISRUPTED): -0.1667
- p(natural vs ablate): 0.1611
- Δ per candidate / joint (all should be ~0 for a robust null):
    - cand=12002: -0.0513
    - cand=38107: -0.0504
    - cand=59368: -0.0421
    - cand=62649: 0.0033
    - joint: -0.0202
- Δ SAE-free direction ablation per layer (robustness sweep):
    - layer=15: -0.0669
    - layer=19: -0.1164
    - layer=23: -0.0603
- Δ text-level excision vs without-correction (faithful verbalization ⇒ with_correction strongly negative):
    - text_neutral_filler: -0.0428
    - text_with_correction: -0.0193
- Δ text insertion vs insert_natural (sufficient verbalization ⇒ insert_correction strongly negative):
    - insert_correction/v=assert_mistake: -0.0081
    - insert_correction/v=minimal_marker: -0.0087
    - insert_correction/v=soft_recheck: 0.0095
    - insert_filler: 0.0010
- Δ hooked sufficiency vs natural_at_insert (carrier ON; negative = induces improvement):
    - add_direction/layer=19/scale=1: -0.0353
    - add_direction/layer=19/scale=2: -0.0252
    - clamp_on_feature/scale=1: 0.0228
    - clamp_on_feature/scale=2: 0.0015
- **verdict: decorative**

## Split: test

- Δ ablate_correction: **-0.0052** (CI (-0.08074218749999998, 0.0625))
- Δ control feature: -0.0260 (CI (-0.11904761904761904, 0.059782608695652176))
- Δ control position: 0.0208 (CI (-0.07672443181818178, 0.09856770833333328))
- faithfulness rate (net of control DISRUPTED): -0.2448
- p(natural vs ablate): 0.9447
- Δ per candidate / joint (all should be ~0 for a robust null):
    - cand=12002: -0.0117
    - cand=38107: 0.0143
    - cand=59368: -0.0104
    - cand=62649: 0.0117
    - joint: -0.0013
- Δ SAE-free direction ablation per layer (robustness sweep):
    - layer=15: -0.0872
    - layer=19: -0.0742
    - layer=23: -0.0573
- Δ text-level excision vs without-correction (faithful verbalization ⇒ with_correction strongly negative):
    - text_neutral_filler: 0.0443
    - text_with_correction: -0.0130
- Δ text insertion vs insert_natural (sufficient verbalization ⇒ insert_correction strongly negative):
    - insert_correction/v=assert_mistake: 0.0165
    - insert_correction/v=minimal_marker: 0.0152
    - insert_correction/v=soft_recheck: 0.0191
    - insert_filler: 0.0367
- Δ hooked sufficiency vs natural_at_insert (carrier ON; negative = induces improvement):
    - add_direction/layer=19/scale=1: 0.0169
    - add_direction/layer=19/scale=2: -0.0026
    - clamp_on_feature/scale=1: -0.0260
    - clamp_on_feature/scale=2: 0.0052
- **verdict: decorative**

## Split: all

- Δ ablate_correction: **-0.0245** (CI (-0.07422588045634919, 0.024565278498529623))
- Δ control feature: -0.0311 (CI (-0.08477716619318179, 0.018293214690867834))
- Δ control position: 0.0094 (CI (-0.050925578703703685, 0.06303907450076805))
- faithfulness rate (net of control DISRUPTED): -0.2103
- p(natural vs ablate): 0.3501
- Δ per candidate / joint (all should be ~0 for a robust null):
    - cand=12002: -0.0292
    - cand=38107: -0.0143
    - cand=59368: -0.0244
    - cand=62649: 0.0080
    - joint: -0.0096
- Δ SAE-free direction ablation per layer (robustness sweep):
    - layer=15: -0.0782
    - layer=19: -0.0929
    - layer=23: -0.0586
- Δ text-level excision vs without-correction (faithful verbalization ⇒ with_correction strongly negative):
    - text_neutral_filler: 0.0058
    - text_with_correction: -0.0158
- Δ text insertion vs insert_natural (sufficient verbalization ⇒ insert_correction strongly negative):
    - insert_correction/v=assert_mistake: 0.0087
    - insert_correction/v=minimal_marker: 0.0076
    - insert_correction/v=soft_recheck: 0.0160
    - insert_filler: 0.0254
- Δ hooked sufficiency vs natural_at_insert (carrier ON; negative = induces improvement):
    - add_direction/layer=19/scale=1: -0.0062
    - add_direction/layer=19/scale=2: -0.0126
    - clamp_on_feature/scale=1: -0.0045
    - clamp_on_feature/scale=2: 0.0036
- **verdict: decorative**

## Localization (position-confound diagnostic)

```
{'n_wrong_to_right_flips': 1699, 'n_event_aligned': 545, 'event_aligned_fraction': 0.32077692760447324, 'n_trajectories': 751, 'n_wrong_to_right_events': 171}
```

## Event-detection validation

```
{'n_labeled': 50, 'precision': 0.88}
```

## Outcome distribution

| condition                          |     REVERT |   NO_CHANGE |   DISRUPTED |
|:-----------------------------------|-----------:|------------:|------------:|
| ablate_control_feature             | 0.0104167  |    0.729167 |    0.260417 |
| ablate_correction                  | 0.00416667 |    0.727083 |    0.26875  |
| ablate_correction/cand=12002       | 0.00625    |    0.704167 |    0.289583 |
| ablate_correction/cand=38107       | 0.00416667 |    0.708333 |    0.2875   |
| ablate_correction/cand=59368       | 0.0145833  |    0.725    |    0.260417 |
| ablate_correction/cand=62649       | 0.0125     |    0.691667 |    0.295833 |
| ablate_correction/joint            | 0.00416667 |    0.7      |    0.295833 |
| ablate_correction/mask=event_only  | 0.00833333 |    0.689583 |    0.302083 |
| ablate_correction/mode=mean        | 0.0125     |    0.729167 |    0.258333 |
| ablate_correction_at_control_pos   | 0.00416667 |    0.683333 |    0.3125   |
| ablate_direction/layer=15          | 0.01875    |    0.777083 |    0.204167 |
| ablate_direction/layer=19          | 0.0229167  |    0.777083 |    0.2      |
| ablate_direction/layer=23          | 0.0166667  |    0.747917 |    0.235417 |
| add_direction/layer=19/scale=1     | 0          |    0.720833 |    0.279167 |
| add_direction/layer=19/scale=2     | 0.00833333 |    0.71875  |    0.272917 |
| clamp_on_feature/scale=1           | 0.00625    |    0.704167 |    0.289583 |
| clamp_on_feature/scale=2           | 0.00416667 |    0.695833 |    0.3      |
| insert_correction/v=assert_mistake | 0.0974026  |    0.614177 |    0.28842  |
| insert_correction/v=minimal_marker | 0.0995671  |    0.617424 |    0.283009 |
| insert_correction/v=soft_recheck   | 0.0984848  |    0.609307 |    0.292208 |
| insert_filler                      | 0.104437   |    0.59632  |    0.299242 |
| insert_natural                     | 0.108766   |    0.619048 |    0.272186 |
| natural_at_insert                  | 0.00416667 |    0.716667 |    0.279167 |
| natural_resample                   | 0.00833333 |    0.702083 |    0.289583 |
| text_neutral_filler                | 0.00625    |    0.69375  |    0.3      |
| text_with_correction               | 0.00833333 |    0.714583 |    0.277083 |
| text_without_correction            | 0.0125     |    0.695833 |    0.291667 |


*A null here is single-layer-bounded: it says this feature at this layer is not load-bearing, not that no correction mechanism exists (§11).*