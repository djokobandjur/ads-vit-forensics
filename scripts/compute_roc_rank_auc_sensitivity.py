#!/usr/bin/env python3
"""
compute_roc_rank_auc_sensitivity.py

Post-processing verification script for the ADS ROC artifact.

Purpose
-------
The primary ROC artifact (ads_roc_v2.json) stores an operating-threshold AUC
computed over a fixed ADS-ratio threshold grid. This script computes a
rank-based sensitivity check from the same archived scores. For each
(PE type, seed, epsilon), the positive side is the single attacked ADS score
for that epsilon and the negative side is the archived clean/benign score set.

Exact single-positive rank AUC is computed as:

    AUC = P(negative < positive) + 0.5 * P(negative == positive)

The script writes a compact JSON summary and prints a table comparing the
rank-based sensitivity AUC to the stored threshold-grid operating AUC.

Typical usage
-------------
    python scripts/compute_roc_rank_auc_sensitivity.py \
        --roc-path data/ads_roc_v2.json \
        --output-path data/ads_roc_rank_auc_sensitivity.json

The script is CPU-only and does not require PyTorch or model checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DEFAULT_PE_TYPES = ["learned", "rope"]
DEFAULT_EPSILONS = ["0.05", "0.1", "0.2"]


def _as_float_list(values: Iterable[Any]) -> List[float]:
    return [float(v) for v in values]


def _sample_std(values: Sequence[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def _fmt_mean_std(values: Sequence[float]) -> str:
    return f"{mean(values):.3f} ± {_sample_std(values):.3f}"


def rank_auc_single_positive(positive: float, negatives: Sequence[float]) -> float:
    """Return exact rank-based AUC for one positive score vs many negatives."""
    if not negatives:
        raise ValueError("Cannot compute AUC with zero negative scores.")
    n_less = sum(1 for x in negatives if x < positive)
    n_equal = sum(1 for x in negatives if x == positive)
    return (n_less + 0.5 * n_equal) / len(negatives)


def sorted_seed_keys(pe_block: Mapping[str, Any]) -> List[str]:
    return sorted(pe_block.keys(), key=lambda s: int(s) if str(s).isdigit() else str(s))


def normalize_eps_key(eps: str, available: Mapping[str, Any]) -> str:
    """Find an epsilon key robustly across JSON string formatting variants."""
    eps_float = float(eps)
    for key in available.keys():
        if math.isclose(float(key), eps_float, rel_tol=0.0, abs_tol=1e-12):
            return key
    raise KeyError(f"epsilon {eps!r} not found; available keys: {sorted(available.keys(), key=float)}")


def compute_rows(
    roc: Mapping[str, Any],
    pe_types: Sequence[str],
    epsilons: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for pe in pe_types:
        if pe not in roc:
            raise KeyError(f"PE type {pe!r} not found in ROC artifact.")

        seeds = sorted_seed_keys(roc[pe])
        for eps in epsilons:
            exact_values: List[float] = []
            stored_values: List[float] = []
            per_seed: List[Dict[str, Any]] = []

            for seed in seeds:
                seed_block = roc[pe][seed]
                attack_ads = seed_block["attack_ads"]
                roc_by_eps = seed_block["roc_by_eps"]
                eps_attack = normalize_eps_key(eps, attack_ads)
                eps_roc = normalize_eps_key(eps, roc_by_eps)

                positive = float(attack_ads[eps_attack])
                negatives = _as_float_list(seed_block["all_negative_scores"])
                exact_auc = rank_auc_single_positive(positive, negatives)
                stored_auc = float(roc_by_eps[eps_roc]["auc"])

                exact_values.append(exact_auc)
                stored_values.append(stored_auc)
                per_seed.append(
                    {
                        "seed": seed,
                        "epsilon": float(eps_attack),
                        "positive_attack_score": positive,
                        "n_positive": 1,
                        "n_negative": len(negatives),
                        "exact_rank_auc": exact_auc,
                        "stored_threshold_auc": stored_auc,
                    }
                )

            # Negative-set composition is normally constant across seeds, but keep
            # it explicit to document the protocol.
            n_negatives = sorted({item["n_negative"] for item in per_seed})

            rows.append(
                {
                    "pe": pe,
                    "epsilon": float(eps),
                    "n_seeds": len(seeds),
                    "n_positive_per_seed": 1,
                    "n_negative_per_seed": n_negatives[0] if len(n_negatives) == 1 else n_negatives,
                    "exact_rank_auc_mean": float(mean(exact_values)),
                    "exact_rank_auc_std": _sample_std(exact_values),
                    "stored_threshold_auc_mean": float(mean(stored_values)),
                    "stored_threshold_auc_std": _sample_std(stored_values),
                    "per_seed": per_seed,
                }
            )

    return rows


def print_table(rows: Sequence[Mapping[str, Any]]) -> None:
    print("PE        eps    exact rank AUC      stored threshold AUC")
    print("---------------------------------------------------------")
    for row in rows:
        exact = f"{row['exact_rank_auc_mean']:.3f} ± {row['exact_rank_auc_std']:.3f}"
        stored = f"{row['stored_threshold_auc_mean']:.3f} ± {row['stored_threshold_auc_std']:.3f}"
        print(f"{row['pe']:<9} {row['epsilon']:<5g} {exact:<18} {stored}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact single-positive rank-AUC sensitivity values from "
            "ads_roc_v2.json and compare them to stored threshold-grid AUCs."
        )
    )
    parser.add_argument(
        "--roc-path",
        default="data/ads_roc_v2.json",
        help="Path to the archived ROC JSON artifact. Default: data/ads_roc_v2.json",
    )
    parser.add_argument(
        "--output-path",
        default="data/ads_roc_rank_auc_sensitivity.json",
        help=(
            "Path for the derived sensitivity-check JSON. "
            "Default: data/ads_roc_rank_auc_sensitivity.json"
        ),
    )
    parser.add_argument(
        "--pe-types",
        nargs="+",
        default=DEFAULT_PE_TYPES,
        help="PE types to process. Default: learned rope",
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        default=DEFAULT_EPSILONS,
        help="Attack budgets to process. Default: 0.05 0.1 0.2",
    )
    args = parser.parse_args()

    roc_path = Path(args.roc_path)
    output_path = Path(args.output_path)

    with roc_path.open("r", encoding="utf-8") as f:
        roc = json.load(f)

    rows = compute_rows(roc, pe_types=args.pe_types, epsilons=args.epsilons)

    payload = {
        "metadata": {
            "source_artifact": str(roc_path),
            "description": (
                "Rank-based single-positive ROC-AUC sensitivity check derived "
                "from the archived ADS ROC scores. The stored ROC table uses a "
                "coarse operating-threshold grid; this file reports exact rank "
                "AUC for one attacked ADS score versus the archived negative "
                "clean/benign scores for each seed and epsilon."
            ),
            "auc_formula": "P(negative < positive) + 0.5 * P(negative == positive)",
            "pe_types": list(args.pe_types),
            "epsilons": [float(e) for e in args.epsilons],
        },
        "results": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print_table(rows)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
