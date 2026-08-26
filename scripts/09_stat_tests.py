#!/usr/bin/env python
"""Stage 09: inferential statistics for a defensible NULL claim.

A non-significant p-value is not evidence of absence. This computes, for the
primary arm and every robustness arm, the statistics a null-result paper needs:

  1. TOST equivalence tests (bootstrap CI-within-margin + paired t versions)
  2. JZS Bayes factors (evidence FOR the null, Rouder et al. 2009)
  3. family-wise max-|Δ| permutation test across ALL ablation arms
  4. power curve / minimum detectable effect at 80% power
  5. localization: chance-alignment baseline + permutation test for the
     event-aligned fraction
  6. DISRUPTED sensitivity (as-wrong / excluded / as-correct) + symmetry test
  7. per-event heterogeneity (overdispersion permutation test)
  8. problem-level clustered bootstrap (chains share problems)

Runs locally against a pulled run dir; no GPU.

  python scripts/09_stat_tests.py --run-dir results/exp1
"""

import re
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import integrate, stats

from src.answers import answers_equal
from src.cli import script_setup
from src.io_utils import read_json, read_jsonl, write_json
from src.metrics import (
    CTRL_FEATURE,
    NATURAL,
    PRIMARY,
    chain_level,
    failed_chain_purity,
    records_to_frame,
    rescore_records,
    split_chains,
)

MARGIN = 0.10          # equivalence margin for TOST (|Δ| < MARGIN)
N_BOOT = 5000
N_PERM = 10000
RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ data loading


def load_records(run_dir: Path) -> tuple[list[dict], dict]:
    records = list(read_jsonl(run_dir / "continuations.jsonl"))
    akp = run_dir / "active_intervention_keys.json"
    if akp.exists():
        active = set(read_json(akp))
        records = [r for r in records if r.get("intervention_key", "") in active]
    # `outcome` was frozen at generation time under whatever answer matcher was
    # current then; re-derive it so a matcher fix reaches the analysis without
    # regenerating 19.8k continuations (see metrics.rescore_records).
    return rescore_records(records)


def paired_chain_deltas(chains: pd.DataFrame, cond: str, col: str = "is_correct",
                        ref: str = NATURAL) -> np.ndarray:
    a = chains[chains.condition == ref].set_index("chain_id")[col]
    b = chains[chains.condition == cond].set_index("chain_id")[col]
    joined = pd.DataFrame({"a": a, "b": b}).dropna()
    return (joined.a - joined.b).to_numpy()


# ------------------------------------------------------------------ 1. TOST


def tost(d: np.ndarray, margin: float = MARGIN) -> dict:
    """Equivalence: reject H0 |Δ| >= margin. t-version + bootstrap-CI version."""
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    t_low = (m + margin) / se     # H0: Δ <= -margin
    t_high = (m - margin) / se    # H0: Δ >= +margin
    p_low = 1 - stats.t.cdf(t_low, n - 1)
    p_high = stats.t.cdf(t_high, n - 1)
    p_tost = max(p_low, p_high)
    boot = np.array([RNG.choice(d, n, replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.quantile(boot, [0.05, 0.95])  # 90% CI <-> alpha=.05 TOST
    return {
        "n_chains": n, "delta_mean": float(m),
        "p_tost": float(p_tost), "equivalent_at_margin": bool(p_tost < 0.05),
        "boot90_ci": [float(lo), float(hi)],
        "boot_ci_within_margin": bool(lo > -MARGIN and hi < MARGIN),
    }


# ------------------------------------------------------------------ 2. JZS BF


def jzs_bf01(d: np.ndarray, r: float = float(np.sqrt(2) / 2)) -> float:
    """Bayes factor FOR the null (BF01), JZS/Cauchy(0, r) prior on effect size
    (Rouder et al. 2009 one-sample). >3 substantial, >10 strong for null."""
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    nu = n - 1

    def integrand(g):
        return ((1 + n * g * r**2) ** -0.5
                * (1 + t**2 / ((1 + n * g * r**2) * nu)) ** (-(nu + 1) / 2)
                * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g)))

    alt, _ = integrate.quad(integrand, 0, np.inf, limit=200)
    null = (1 + t**2 / nu) ** (-(nu + 1) / 2)
    return float(null / alt)


# ------------------------------------------ 3. family-wise max-|Δ| permutation


def maxT_permutation(chains: pd.DataFrame, arms: list[str], ref: str = NATURAL) -> dict:
    """Within each chain, condition labels are exchangeable under the global
    null; max |Δ| over ALL arms gives one family-wise p-value. Restricted to
    the family's own conditions BEFORE pivoting — otherwise a chain missing
    any unrelated condition (e.g. an insertion arm at a different truncation)
    would be dropped from this family's test too. `ref` lets the insertion
    families use their own in-family reference arm."""
    fam = chains[chains.condition.isin([ref] + list(arms))]
    wide = fam.pivot(index="chain_id", columns="condition", values="is_correct").dropna()
    conds = list(wide.columns)
    if ref not in conds:
        return {"error": f"reference {ref} absent", "n_arms": 0}
    nat_i = conds.index(ref)
    arm_idx = [conds.index(a) for a in arms if a in conds]
    X = wide.to_numpy()

    def max_delta(M):
        return max(abs((M[:, nat_i] - M[:, j]).mean()) for j in arm_idx)

    observed = max_delta(X)
    count = 0
    for _ in range(N_PERM):
        P = np.array([row[RNG.permutation(len(conds))] for row in X])
        if max_delta(P) >= observed - 1e-12:
            count += 1
    return {
        "observed_max_abs_delta": float(observed),
        "p_familywise": float((count + 1) / (N_PERM + 1)),
        "n_arms": len(arm_idx), "n_chains": int(X.shape[0]),
    }


# ------------------------------------------------------- 4. power / MDE


def power_curve(d: np.ndarray, effects=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
                n_sim: int = 800, n_signflip: int = 1000) -> dict:
    """Monte-Carlo power of the sign-flip test if a true effect δ existed:
    resample chains, shift by δ, test at α=.05. MDE = smallest δ with ≥80%."""
    n = len(d)
    centered = d - d.mean()  # null-calibrated noise from the real data

    def signflip_p(x):
        obs = abs(x.mean())
        flips = RNG.choice([-1.0, 1.0], size=(n_signflip, len(x)))
        return (np.sum(np.abs((flips * x).mean(axis=1)) >= obs - 1e-12) + 1) / (n_signflip + 1)

    curve = {}
    for eff in effects:
        hits = 0
        for _ in range(n_sim):
            x = RNG.choice(centered, n, replace=True) + eff
            if signflip_p(x) < 0.05:
                hits += 1
        curve[eff] = hits / n_sim
    mde = next((e for e, p in curve.items() if p >= 0.8), None)
    return {"power_by_effect": {str(k): float(v) for k, v in curve.items()},
            "mde_80pct": mde}


# --------------------------------------------------- sufficiency-family helpers

# arms that live at the INSERTION point (stages 11/12) — never compared
# against natural_resample (different truncation), so they are excluded from
# every natural-referenced analysis and get their own within-family sections
INSERTION_FAMILY = ("insert_", "clamp_on", "add_direction/", "natural_at_insert")


def signflip_p(d: np.ndarray) -> float:
    """Two-sided sign-flip permutation p on paired deltas (directional
    evidence for the sufficiency arms, where a real effect is the hypothesis
    of interest rather than the thing being equivalence-tested away)."""
    obs = abs(d.mean())
    flips = RNG.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    return float((np.sum(np.abs((flips * d).mean(axis=1)) >= obs - 1e-12) + 1)
                 / (N_PERM + 1))


def signflip_p_helps(d: np.ndarray) -> float:
    """One-sided sign-flip p for 'the insertion HELPS' (Δ = ref − arm, so
    help = negative): P(perm mean <= observed mean)."""
    obs = d.mean()
    flips = RNG.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    return float((np.sum((flips * d).mean(axis=1) <= obs + 1e-12) + 1) / (N_PERM + 1))


def family_stats(sub: pd.DataFrame, chains: pd.DataFrame, ref: str,
                 prefixes: tuple[str, ...]) -> Optional[dict]:
    """TOST + BF01 + two-sided sign-flip p for every family arm against the
    family's OWN reference condition (same backend, same truncation, shared
    seeds). Sign: Δ = ref − arm, so an arm that HELPS is negative."""
    if ref not in set(sub.condition.unique()):
        return None
    out = {}
    for arm in sorted(c for c in sub.condition.unique()
                      if c.startswith(prefixes) and c != ref):
        d = paired_chain_deltas(chains, arm, ref=ref)
        if len(d) >= 5:
            out[arm] = {**tost(d), "bf01_jzs": jzs_bf01(d),
                        "p_two_sided": signflip_p(d),
                        "p_one_sided_helps": signflip_p_helps(d)}
    return out or None


def marker_induction(df: pd.DataFrame) -> Optional[dict]:
    """Induced-verbalization rates: fraction of continuations whose text
    verbalizes a correction marker near the start, per sufficiency arm.
    Mechanistic sufficiency for the SIGNAL predicts clamp_on/add_direction
    well above natural_at_insert."""
    if "meta" not in df.columns:
        return None
    fam = df[df.condition.str.startswith(INSERTION_FAMILY)].copy()
    if fam.empty:
        return None
    fam["marker"] = fam["meta"].apply(
        lambda m: m.get("marker_in_continuation") if isinstance(m, dict) else None)
    fam["pop"] = fam["meta"].apply(
        lambda m: m.get("population", "") if isinstance(m, dict) else "")
    fam = fam.dropna(subset=["marker"])
    if fam.empty:
        return None
    return {
        (f"{cond}[{p}]" if p else cond): {"rate": float(g.marker.mean()), "n": int(len(g))}
        for (cond, p), g in fam.groupby(["condition", "pop"])
    }


# ------------------------------------------------ 5. localization chance level


def localization_test(run_dir: Path, align_window: int = 8) -> dict:
    """Are wrong->right flips co-located with correction events more than
    chance? Two nulls, because the answer is sensitive to which one you pick.

    A flip is observed between consecutive probe points, so the unit is the
    probe INTERVAL. The probe grid is `stride` tokens PLUS every event onset
    and offset, which chops the neighbourhood of a marker into short intervals:
    aligned intervals are about half the token length of unaligned ones. That
    makes the choice of null a real modelling decision:

      interval-uniform   each of a chain's intervals is equally likely to hold
                         a flip. Ignores that a longer interval has more room
                         for the intended answer to move.
      token-uniform      a flip lands uniformly in TOKEN time, so an interval's
                         probability is proportional to its length. The natural
                         null for a process unfolding in token time.

    Both are reported. Each p-value is computed against ITS OWN permutation
    distribution's centre — an earlier version compared the observed value to a
    separately-computed pooled interval fraction whose value (0.314) differed
    from the permutation mean (0.301) by more than the effect under test, which
    made the test unable to reject by construction.

    Probe `correct` flags are recomputed from the stored answer against the
    chain's gold with the current matcher, so this estimand tracks
    answers.normalize_answer rather than whatever was current at probe time.
    """
    events_by_chain: dict[str, list] = {}
    for d in read_jsonl(run_dir / "events.jsonl"):
        events_by_chain.setdefault(d["chain_id"], []).append(d["onset"])
    gold_by_chain = {d["chain_id"]: d["gold_answer"]
                     for d in read_jsonl(run_dir / "chains.jsonl")}

    per_chain = []  # (n_w2r_flips, n_aligned, aligned_flags, interval_lengths)
    for t in read_jsonl(run_dir / "trajectories.jsonl"):
        pts = t["points"]
        onsets = events_by_chain.get(t["chain_id"], [])
        gold = gold_by_chain.get(t["chain_id"])
        flags, lengths, w2r, aligned = [], [], 0, 0
        for a, b in zip(pts, pts[1:]):
            is_aligned = any(a["position"] - align_window < o <= b["position"] for o in onsets)
            flags.append(is_aligned)
            lengths.append(max(b["position"] - a["position"], 1))
            a_correct = answers_equal(a["answer"], gold)
            b_correct = answers_equal(b["answer"], gold)
            if (a["answer"] is not None and not a_correct and b_correct
                    and not answers_equal(a["answer"], b["answer"])):
                w2r += 1
                aligned += is_aligned
        if flags:
            per_chain.append((w2r, aligned, np.array(flags, dtype=bool),
                              np.array(lengths, dtype=float)))

    tot_w2r = sum(c[0] for c in per_chain)
    tot_aligned = sum(c[1] for c in per_chain)
    observed = tot_aligned / tot_w2r
    all_flags = np.concatenate([c[2] for c in per_chain])
    all_lens = np.concatenate([c[3] for c in per_chain])

    def permute(weighted: bool, n_perm: int) -> np.ndarray:
        out = np.zeros(n_perm)
        for i in range(n_perm):
            tot = 0
            for w2r, _, flags, lens in per_chain:
                if not (w2r and len(flags)):
                    continue
                k = min(w2r, len(flags))
                if weighted and k < len(flags):
                    pick = RNG.choice(len(flags), k, replace=False, p=lens / lens.sum())
                else:
                    pick = RNG.choice(len(flags), k, replace=False)
                tot += flags[pick].sum()
            out[i] = tot / tot_w2r
        return out

    def summarize(perm: np.ndarray) -> dict:
        centre, sd = float(perm.mean()), float(perm.std())
        p = float((np.sum(np.abs(perm - centre) >= abs(observed - centre) - 1e-12) + 1)
                  / (len(perm) + 1))
        return {
            "expected_aligned_fraction": centre,
            "sd": sd,
            "z": float((observed - centre) / sd) if sd > 0 else None,
            "enrichment": float(observed / centre) if centre > 0 else None,
            "p_two_sided": p,
            "p_one_sided_above": float((np.sum(perm >= observed - 1e-12) + 1) / (len(perm) + 1)),
        }

    return {
        "observed_aligned_fraction": float(observed),
        "n_wrong_to_right_flips": int(tot_w2r),
        "n_aligned": int(tot_aligned),
        "n_intervals": int(len(all_flags)),
        "aligned_interval_token_share": float(all_lens[all_flags].sum() / all_lens.sum()),
        "mean_interval_len_aligned": float(all_lens[all_flags].mean()),
        "mean_interval_len_unaligned": float(all_lens[~all_flags].mean()),
        "null_interval_uniform": summarize(permute(False, N_PERM)),
        "null_token_uniform": summarize(permute(True, 2000)),
        "note": ("aligned intervals are shorter because the probe grid inserts "
                 "event boundaries; the token-uniform null accounts for that and "
                 "is the primary one. The finer grid near events also resolves "
                 "flips a uniform grid would merge, which inflates alignment "
                 "under either null."),
    }


# ------------------------------------------- 6. DISRUPTED sensitivity/symmetry


def disrupted_sensitivity(df: pd.DataFrame) -> dict:
    out = {}
    for name, frame in {
        "disrupted_as_wrong (primary)": df.assign(v=df.outcome == "NO_CHANGE"),
        "disrupted_excluded": df[df.outcome != "DISRUPTED"].assign(
            v=lambda x: x.outcome == "NO_CHANGE"),
        "disrupted_as_correct (extreme bound)": df.assign(
            v=df.outcome.isin(["NO_CHANGE", "DISRUPTED"])),
    }.items():
        ev = frame.groupby(["chain_id", "event_id", "condition"])["v"].mean().reset_index()
        ch = ev.groupby(["chain_id", "condition"])["v"].mean().reset_index()
        a = ch[ch.condition == NATURAL].set_index("chain_id")["v"]
        b = ch[ch.condition == PRIMARY].set_index("chain_id")["v"]
        j = pd.DataFrame({"a": a, "b": b}).dropna()
        out[name] = float((j.a - j.b).mean())
    # symmetry: paired sign-flip on per-chain disruption rates
    dfd = df.assign(v=df.outcome == "DISRUPTED")
    ev = dfd.groupby(["chain_id", "event_id", "condition"])["v"].mean().reset_index()
    ch = ev.groupby(["chain_id", "condition"])["v"].mean().reset_index()
    a = ch[ch.condition == NATURAL].set_index("chain_id")["v"]
    b = ch[ch.condition == PRIMARY].set_index("chain_id")["v"]
    d = (pd.DataFrame({"a": a, "b": b}).dropna().eval("a-b")).to_numpy()
    obs = abs(d.mean())
    flips = RNG.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    p = float((np.sum(np.abs((flips * d).mean(axis=1)) >= obs - 1e-12) + 1) / (N_PERM + 1))
    out["disruption_rate_diff_natural_minus_ablate"] = float(d.mean())
    out["disruption_symmetry_p"] = p
    return out


# ------------------------------------------------- 7. per-event heterogeneity


def heterogeneity_test(df: pd.DataFrame) -> dict:
    """Does variance of per-event effects exceed condition-shuffling noise?
    (A hidden faithful subpopulation would inflate it.)"""
    sub = df[df.condition.isin([NATURAL, PRIMARY])]
    piv = (sub.groupby(["event_id", "condition"])["is_correct"].mean()
              .unstack().dropna())
    d_e = (piv[NATURAL] - piv[PRIMARY]).to_numpy()
    observed_var = float(np.var(d_e))

    raw = sub.groupby(["event_id", "condition"])["is_correct"].apply(list).unstack().dropna()
    perm_vars = np.zeros(2000)
    for i in range(2000):
        ds = []
        for _, row in raw.iterrows():
            pool = np.array(row[NATURAL] + row[PRIMARY])
            RNG.shuffle(pool)
            k = len(row[NATURAL])
            ds.append(pool[:k].mean() - pool[k:].mean())
        perm_vars[i] = np.var(ds)
    p = float((np.sum(perm_vars >= observed_var - 1e-12) + 1) / 2001)
    return {"observed_var_event_effects": observed_var,
            "perm_var_mean": float(perm_vars.mean()),
            "p_overdispersion": p, "n_events": int(len(d_e)),
            "max_event_effect": float(d_e.max()), "min_event_effect": float(d_e.min())}


# ---------------------------------------------- 8. problem-clustered bootstrap


def clustered_bootstrap(df: pd.DataFrame, cond: str) -> dict:
    """Resample PROBLEMS (chains share problems, 4 chains/problem)."""
    df = df.copy()
    df["problem_id"] = df["meta"].apply(
        lambda m: m.get("problem_id") if isinstance(m, dict) else None)
    ev = df.groupby(["problem_id", "chain_id", "event_id", "condition"])["is_correct"].mean().reset_index()
    ch = ev.groupby(["problem_id", "chain_id", "condition"])["is_correct"].mean().reset_index()
    # only problems carrying BOTH compared conditions contribute to the delta;
    # resampling the others just adds empty draws and widens the interval for
    # the wrong reason, so restrict the resampling frame to the informative set
    contributing = set(ch[ch.condition == NATURAL].problem_id) & set(ch[ch.condition == cond].problem_id)
    ch = ch[ch.problem_id.isin(contributing)]
    probs = ch.problem_id.unique()
    by_prob = {p: g for p, g in ch.groupby("problem_id")}

    def delta(frame):
        a = frame[frame.condition == NATURAL].groupby("chain_id")["is_correct"].mean()
        b = frame[frame.condition == cond].groupby("chain_id")["is_correct"].mean()
        j = pd.DataFrame({"a": a, "b": b}).dropna()
        return (j.a - j.b).mean() if len(j) else np.nan

    boot = []
    for _ in range(2000):
        sample = RNG.choice(probs, len(probs), replace=True)
        frame = pd.concat([by_prob[p].assign(chain_id=by_prob[p].chain_id + f"#{i}")
                           for i, p in enumerate(sample)])
        v = delta(frame)
        if not np.isnan(v):
            boot.append(v)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"point": float(delta(ch)), "ci95_problem_clustered": [float(lo), float(hi)],
            "n_problems": int(len(probs))}


# --------------------------------------- 9. did the intervention do anything?


def seed_matched_divergence(records: list[dict], pairs: Sequence[tuple[str, str]]) -> dict:
    """Fraction of continuations whose TEXT differs from the seed-matched
    reference continuation of the same (chain, event).

    This is the empirical answer to "did the intervention actually fire?".
    A clamp that never changed the residual stream would decode bit-identical
    text under a shared seed, so a high divergence rate is direct evidence the
    manipulation bit; a high IDENTITY rate flags an arm whose equivalence is
    partly guaranteed by construction rather than measured.
    """
    by_key = {(r["chain_id"], r["event_id"], r["condition"], r["seed"]): r.get("text", "")
              for r in records}
    out = {}
    for arm, ref in pairs:
        n = ident = 0
        prefixes = []
        for (cid, eid, cond, seed), text in by_key.items():
            if cond != arm:
                continue
            rt = by_key.get((cid, eid, ref, seed))
            if rt is None:
                continue
            n += 1
            if text == rt:
                ident += 1
            else:
                shared = 0
                for x, y in zip(text, rt):
                    if x != y:
                        break
                    shared += 1
                prefixes.append(shared)
        if n:
            out[arm] = {
                "reference": ref, "n": n,
                "identical": ident, "identical_rate": ident / n,
                "divergence_rate": 1 - ident / n,
                "median_shared_prefix_chars": int(np.median(prefixes)) if prefixes else None,
            }
    return out


# ------------------------------- 10. second-order (answer-distribution) nulls


def answer_distribution_stats(df: pd.DataFrame, arms: Sequence[str],
                              ref: str = NATURAL) -> dict:
    """Per-event statistics of the empirical answer DISTRIBUTION over the K
    shared seeds, paired arm-vs-reference.

    Accuracy is an argmax statistic: an intervention could reshape the model's
    distribution over answers — its spread, its mass on the pre-correction
    answer — while leaving the modal answer intact, and a mean-accuracy delta
    would not see it. These are the second-order estimands, and the K seeds
    per (event, condition) already pay for them.
    """
    from src.answers import normalize_answer

    def key(a):
        return "<none>" if a is None or (isinstance(a, float)) else \
            re.sub(r"\s+", "", normalize_answer(a)).lower()

    sub = df[df.condition.isin([ref] + list(arms))].copy()
    sub["ans"] = sub.final_answer.map(key)
    sub["gold_k"] = sub["meta"].map(lambda m: key(m.get("gold_answer")) if isinstance(m, dict) else None)
    sub["pre_k"] = sub["meta"].map(lambda m: key(m.get("pre_answer")) if isinstance(m, dict) else None)

    rows = []
    for (ev, cond), g in sub.groupby(["event_id", "condition"]):
        counts = g.ans.value_counts(normalize=True)
        rows.append({
            "event_id": ev, "condition": cond,
            "entropy": float(-(counts * np.log2(counts)).sum()),
            "mode_share": float(counts.max()),
            "n_distinct": int(g.ans.nunique()),
            "mass_gold": float((g.ans == g.gold_k).mean()),
            "mass_pre": float((g.ans == g.pre_k).mean()),
            "mass_unparsed": float((g.ans == "<none>").mean()),
        })
    E = pd.DataFrame(rows)
    cols = ["entropy", "mode_share", "n_distinct", "mass_gold", "mass_pre", "mass_unparsed"]
    piv = E.pivot(index="event_id", columns="condition")
    out = {"reference": ref, "reference_means": {c: float(E[E.condition == ref][c].mean()) for c in cols},
           "arms": {}}
    for arm in arms:
        if (cols[0], arm) not in piv.columns:
            continue
        stat = {}
        for c in cols:
            d = (piv[(c, arm)] - piv[(c, ref)]).dropna().to_numpy()
            if len(d) < 5:
                continue
            stat[c] = {
                "arm_mean": float(piv[(c, arm)].mean()),
                "delta_arm_minus_ref": float(d.mean()),
                "p_signflip": signflip_p(d),
                "n_events": int(len(d)),
            }
        out["arms"][arm] = stat
    return out


# ----------------------------------------------------------------------- main


def main() -> None:
    cfg, cfg_dict, run_dir = script_setup("09_stat_tests")
    records, rescore_summary = load_records(run_dir)
    purity = failed_chain_purity(records)
    df = records_to_frame(records)
    print(f"{len(df)} active continuations, {df.condition.nunique()} conditions")
    print(f"rescored under the current answer matcher: {rescore_summary['n_rescored']} "
          f"{rescore_summary['transitions']}")
    print(f"failed_chain population: {purity['n_truly_failed']}/{purity['n_chains']} truly failed "
          f"({purity['n_contaminated']} admitted by matcher false negatives)")

    split = split_chains(sorted(df.chain_id.unique()),
                         cfg.experiment.splits.pilot_fraction,
                         cfg.experiment.splits.split_seed)
    # insertion-family arms live at a different truncation than natural_resample
    # — they are analyzed within-family only (see family_stats), never here
    arms = sorted(c for c in df.condition.unique()
                  if c != NATURAL and not c.startswith(INSERTION_FAMILY))
    results: dict = {"margin": MARGIN, "splits": {}}

    for split_name in ("test", "all"):
        sub = df if split_name == "all" else df[df.chain_id.map(split) == split_name]
        chains = chain_level(sub)
        per_arm = {}
        for arm in arms:
            d = paired_chain_deltas(chains, arm)
            if len(d) < 5:
                continue
            per_arm[arm] = {**tost(d), "bf01_jzs": jzs_bf01(d)}
        # two families: the SAE-feature null claim, and the (exploratory)
        # direction-ablation arms — pooling them would let the direction
        # arms' real anti-faithful effect masquerade as "some ablation works"
        sae_arms = [a for a in arms
                    if not a.startswith(("ablate_direction", "text_"))]
        dir_arms = [a for a in arms if a.startswith("ablate_direction")]
        results["splits"][split_name] = {
            "per_arm": per_arm,
            "familywise_maxT_all": maxT_permutation(chains, arms),
            "familywise_maxT_sae_family": maxT_permutation(chains, sae_arms),
            "familywise_maxT_direction_family": maxT_permutation(chains, dir_arms)
            if dir_arms else None,
            "power": power_curve(paired_chain_deltas(chains, PRIMARY)),
        }

        # text-level excision: inference WITHIN the text experiment (all its
        # arms share one backend + seeds); reference = without-correction.
        # Faithful verbalization predicts with_correction Δ strongly NEGATIVE.
        text_ref = "text_without_correction"
        if text_ref in set(sub.condition.unique()):
            text_stats = {}
            for arm in sorted(c for c in sub.condition.unique()
                              if c.startswith("text_") and c != text_ref):
                d = paired_chain_deltas(chains, arm, ref=text_ref)
                if len(d) >= 5:
                    text_stats[arm] = {**tost(d), "bf01_jzs": jzs_bf01(d)}
            results["splits"][split_name]["text_excision"] = text_stats

        # sufficiency families (stages 11/12): each against its own in-family
        # reference. Sign: Δ = ref − arm; a sufficient verbalization/carrier
        # would make insert_correction / clamp_on strongly NEGATIVE.
        ins = family_stats(sub, chains, "insert_natural", ("insert_",))
        if ins:
            results["splits"][split_name]["text_insertion"] = ins
            ins_arms = sorted(ins.keys())
            results["splits"][split_name]["familywise_maxT_insertion"] = (
                maxT_permutation(chains, ins_arms, ref="insert_natural"))
            # per population: wrong_point chains self-corrected elsewhere
            # (high base rescue = ceiling); failed_chain chains never did
            # (no ceiling — the clean sufficiency population)
            pop = sub["meta"].apply(
                lambda m: m.get("population") if isinstance(m, dict) else None)
            for p in ("wrong_point", "failed_chain"):
                sub_p = sub[pop == p]
                if sub_p.empty:
                    continue
                stats_p = family_stats(sub_p, chain_level(sub_p),
                                       "insert_natural", ("insert_",))
                if stats_p:
                    results["splits"][split_name][f"text_insertion_{p}"] = stats_p
                if p == "failed_chain":
                    # the population's selection rule ("parses AND wrong") admits
                    # every answer-matcher false negative, so the confirmatory
                    # estimand runs on the chains that are actually never-rescued
                    clean = sub_p[sub_p.chain_id.isin(purity["clean_chain_ids"])]
                    if not clean.empty:
                        stats_c = family_stats(clean, chain_level(clean),
                                               "insert_natural", ("insert_",))
                        if stats_c:
                            results["splits"][split_name]["text_insertion_failed_chain_clean"] = stats_c
                        results["splits"][split_name]["failed_chain_clean_accuracy"] = {
                            c: float(g.is_correct.mean())
                            for c, g in chain_level(clean).groupby("condition")
                        }
        suf = family_stats(sub, chains, "natural_at_insert",
                           ("clamp_on", "add_direction/"))
        if suf:
            results["splits"][split_name]["hooked_sufficiency"] = suf
            results["splits"][split_name]["familywise_maxT_hooked_sufficiency"] = (
                maxT_permutation(chains, sorted(suf.keys()), ref="natural_at_insert"))

    results["rescoring"] = rescore_summary
    # keep the id list: figures and any re-analysis need to reproduce the
    # matcher-corrected population without re-deriving it
    results["failed_chain_purity"] = purity
    results["marker_induction"] = marker_induction(df)
    results["intervention_divergence"] = seed_matched_divergence(
        records,
        [(a, NATURAL) for a in sorted(df.condition.unique())
         if a not in (NATURAL,) and not a.startswith(INSERTION_FAMILY)]
        + [(a, "natural_at_insert") for a in sorted(df.condition.unique())
           if a.startswith(("clamp_on", "add_direction/"))],
    )
    results["answer_distribution"] = answer_distribution_stats(
        df, [PRIMARY, "ablate_correction/joint", CTRL_FEATURE, "ablate_direction/layer=19"])
    results["localization"] = localization_test(
        run_dir, cfg.experiment.trajectory.align_window_tokens)
    results["disrupted_sensitivity"] = disrupted_sensitivity(df)
    results["heterogeneity_natural_vs_ablate"] = heterogeneity_test(df)
    results["problem_clustered_bootstrap_primary"] = clustered_bootstrap(df, PRIMARY)

    out = run_dir / "analysis" / "stat_tests.json"
    write_json(out, results)
    print(f"-> {out}")

    # concise console digest
    t = results["splits"]["test"]
    prim = t["per_arm"].get(PRIMARY, {})
    print("\n=== DIGEST (test split) ===")
    print(f"primary TOST p={prim.get('p_tost'):.4g} equivalent={prim.get('equivalent_at_margin')} "
          f"BF01={prim.get('bf01_jzs'):.1f}")
    eq = sum(1 for a in t["per_arm"].values() if a["equivalent_at_margin"])
    print(f"arms equivalent at |Δ|<{MARGIN}: {eq}/{len(t['per_arm'])}")
    print(f"family-wise max|Δ| p: all={t['familywise_maxT_all']['p_familywise']:.4g} "
          f"sae={t['familywise_maxT_sae_family']['p_familywise']:.4g} "
          f"direction={t['familywise_maxT_direction_family']['p_familywise']:.4g}")
    print(f"MDE(80% power)={t['power']['mde_80pct']}")
    for arm, s in (t.get("text_excision") or {}).items():
        print(f"text excision {arm}: Δ={s['delta_mean']:+.4f} "
              f"TOSTp={s['p_tost']:.3g} BF01={s['bf01_jzs']:.1f}")
    for section in ("text_insertion", "text_insertion_wrong_point",
                    "text_insertion_failed_chain", "hooked_sufficiency"):
        for arm, s in (t.get(section) or {}).items():
            print(f"{section} {arm}: Δ={s['delta_mean']:+.4f} (neg=helps) "
                  f"p1={s['p_one_sided_helps']:.3g} p2={s['p_two_sided']:.3g} "
                  f"TOSTp={s['p_tost']:.3g} BF01={s['bf01_jzs']:.1f}")
    if results.get("marker_induction"):
        print("marker induction (verbalizes near start):")
        for cond, r in sorted(results["marker_induction"].items()):
            print(f"    {cond}: {r['rate']:.3f} (n={r['n']})")
    loc = results["localization"]
    print(f"localization: observed={loc['observed_aligned_fraction']:.3f} of "
          f"{loc['n_wrong_to_right_flips']} flips; aligned intervals hold "
          f"{loc['aligned_interval_token_share']:.3f} of tokens")
    for null_name in ("null_interval_uniform", "null_token_uniform"):
        n = loc[null_name]
        print(f"    {null_name:22s} expected={n['expected_aligned_fraction']:.3f} "
              f"enrichment={n['enrichment']:.2f}x z={n['z']:.2f} p={n['p_two_sided']:.4g}")
    print("intervention divergence (did the clamp fire?):")
    for arm, d in sorted(results["intervention_divergence"].items()):
        print(f"    {arm}: diverged {d['divergence_rate']:.3f} of {d['n']} "
              f"(identical {d['identical']})")
    print(f"disrupted sensitivity: {results['disrupted_sensitivity']}")
    print(f"heterogeneity p={results['heterogeneity_natural_vs_ablate']['p_overdispersion']:.4g}")
    print(f"problem-clustered CI: {results['problem_clustered_bootstrap_primary']}")


if __name__ == "__main__":
    main()
