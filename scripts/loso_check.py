#!/usr/bin/env python3
"""
loso_check.py -- Leave-one-seed-out (LOSO) threshold calibration check.

Addresses the calibration/evaluation coupling objection: the conservative
detection thresholds (3.0x Learned, 6.2x RoPE) are derived from the worst-case
benign-shift ratio across the same six seeds on which evasion is evaluated.
This script recomputes every evasion count under LOSO calibration: for each
held-out seed, the threshold is set from the worst benign ratio of the
remaining five seeds, rounded upward to one decimal (the same rounding
convention as the paper: 2.95 -> 3.0, 6.17 -> 6.2).

Inputs (from --data-dir):
    ads_roc_v2.json       per-seed clean noise floors and benign-shift ADS
    ads_ref_evasion.json  reference-set evasion sweep (ads_ref_ratio, ads_hold_ratio)
    ads_adaptive.json     adaptive-attacker sweep (ads_ratio)

Usage:
    python scripts/loso_check.py --data-dir data

NOTE: Evasion counts are only as meaningful as the attack that produced the
input JSONs. If ads_ref_evasion.json / ads_adaptive.json are regenerated
(e.g., after correcting the PGD sign convention in ads_ref_evasion.py /
ads_adaptive.py), rerun this script on the new files.

Convention notes:
  - Per-seed worst benign ratio = max over benign transforms of
    (benign ADS / per-seed clean noise-floor mean). This is the seed-wise
    quantity behind the paper's threshold rounding; the benign *table* in the
    paper instead reports ratios of seed-pooled means, which is a different
    (smaller) summary of the same data.
  - "Evades" = ADS ratio strictly below threshold.
"""

import argparse
import json
import math
import os
import statistics as st

PE_TYPES = ["learned", "rope"]


def round_up_1dec(x: float) -> float:
    return math.ceil(x * 10.0) / 10.0


def load(data_dir: str, name: str):
    path = os.path.join(data_dir, name)
    with open(path) as f:
        return json.load(f)


def per_seed_worst_benign(roc: dict) -> dict:
    """worst[pe][seed] = max_t benign_ads[t] / mean(clean_noise_floor)."""
    worst = {}
    for pe in PE_TYPES:
        worst[pe] = {}
        for seed, d in roc[pe].items():
            base = st.mean(d["clean_noise_floor"]["scores"])
            worst[pe][seed] = max(v / base for v in d["benign_ads"].values())
    return worst


def loso_taus(worst_pe: dict) -> dict:
    """LOSO threshold per held-out seed (round-up of the other seeds' max)."""
    return {
        s: round_up_1dec(max(v for o, v in worst_pe.items() if o != s))
        for s in worst_pe
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data", help="Directory with the JSON files")
    args = ap.parse_args()

    roc = load(args.data_dir, "ads_roc_v2.json")
    ref = load(args.data_dir, "ads_ref_evasion.json")
    ada = load(args.data_dir, "ads_adaptive.json")

    worst = per_seed_worst_benign(roc)

    for pe in PE_TYPES:
        tau_full = round_up_1dec(max(worst[pe].values()))
        taus = loso_taus(worst[pe])

        print(f"\n=== {pe.upper()} ===")
        print(f"full-sample tau = {tau_full}x")
        for s in worst[pe]:
            print(f"  seed {s:>5}: worst benign = {worst[pe][s]:.2f}x  ->  LOSO tau = {taus[s]}x")

        print("\n  reference-set evasion (counts over 6 seeds x all lambda):")
        for eps in sorted(next(iter(ref[pe].values())).keys(), key=float):
            n = full = l_ref = l_hold = l_both = 0
            for s, per_eps in ref[pe].items():
                for lam, e in per_eps[eps].items():
                    n += 1
                    full += e["ads_ref_ratio"] < tau_full
                    r = e["ads_ref_ratio"] < taus[s]
                    h = e["ads_hold_ratio"] < taus[s]
                    l_ref += r
                    l_hold += h
                    l_both += r and h
            print(
                f"    eps={eps}: full tau -> {full}/{n} (ref) | "
                f"LOSO -> {l_ref}/{n} ref, {l_hold}/{n} hold, {l_both}/{n} both"
            )

        print("\n  adaptive attacker (counts over 6 seeds x all lambda):")
        for eps in sorted(next(iter(ada[pe].values())).keys(), key=float):
            n = full = loso = 0
            for s, per_eps in ada[pe].items():
                for lam, e in per_eps[eps].items():
                    n += 1
                    full += e["ads_ratio"] < tau_full
                    loso += e["ads_ratio"] < taus[s]
            print(f"    eps={eps}: full tau -> {full}/{n} | LOSO -> {loso}/{n}")

    print(
        "\nInterpretation: identical or lower LOSO counts mean the "
        "calibration/evaluation seed coupling does not drive any reported "
        "evasion result."
    )


if __name__ == "__main__":
    main()
