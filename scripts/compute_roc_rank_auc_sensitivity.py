#!/usr/bin/env python3
"""
compute_roc_rank_auc_sensitivity.py

CPU-only post-processing script for the ADS benign-aware ROC artifact.

Purpose
-------
The main manuscript reports standard rank-based AUC values for the benign-aware
ROC table. This script derives those rank-based values from the archived
ads_roc_v2.json scores. For each (PE type, seed, epsilon), the positive side is
one attacked ADS score and the negative side is the archived clean/benign score
set.

The same output also records the archived threshold-grid trapezoidal values from
ads_roc_v2.json -> roc_by_eps[eps].auc as provenance. Those archived values are
not used as the main AUC estimator.

Rank-based single-positive AUC is computed as:

    AUC = P(negative < positive) + 0.5 * P(negative == positive)

Typical usage
-------------
    python scripts/compute_roc_rank_auc_sensitivity.py \
        --roc-path data/ads_roc_v2.json \
        --output-path data/ads_roc_rank_auc_sensitivity.json

The default output contains the compact fields consumed by reproduce.py:

    pe, epsilon, exact_mean, exact_std, stored_mean, stored_std, n_seeds

plus descriptive documentation fields (exact_estimator,
stored_provenance_quantity, stored_provenance_field, n_positive_per_seed,
n_negative_per_seed). reproduce.py reads only the compact fields.

where exact_* are rank-based AUC summaries and stored_* are the archived
threshold-grid trapezoidal provenance summaries.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DEFAULT_PE_TYPES = ["learned", "rope"]
DEFAULT_EPSILONS = ["0.05", "0.1", "0.2", "0.5"]


def _as_float_list(values: Iterable[Any]) -> List[float]:
    return [float(v) for v in values]


def _sample_std(values: Sequence[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


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
    """Find an epsilon key robustly across JSON string-format variants."""
    eps_float = float(eps)
    for key in available.keys():
        if math.isclose(float(key), eps_float, rel_tol=0.0, abs_tol=1e-12):
            return key
    raise KeyError(f"epsilon {eps!r} not found; available keys: {sorted(available.keys(), key=float)}")


def compute_rows(
    roc: Mapping[str, Any],
    pe_types: Sequence[str],
    epsilons: Sequence[str],
    include_per_seed: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for pe in pe_types:
        if pe not in roc:
            raise KeyError(f"PE type {pe!r} not found in ROC artifact.")

        seeds = sorted_seed_keys(roc[pe])
        for eps in epsilons:
            rank_values: List[float] = []
            threshold_grid_values: List[float] = []
            per_seed: List[Dict[str, Any]] = []

            for seed in seeds:
                seed_block = roc[pe][seed]
                attack_ads = seed_block["attack_ads"]
                roc_by_eps = seed_block["roc_by_eps"]
                eps_attack = normalize_eps_key(eps, attack_ads)
                eps_roc = normalize_eps_key(eps, roc_by_eps)

                positive = float(attack_ads[eps_attack])
                negatives = _as_float_list(seed_block["all_negative_scores"])
                rank_auc = rank_auc_single_positive(positive, negatives)
                threshold_grid_value = float(roc_by_eps[eps_roc]["auc"])

                rank_values.append(rank_auc)
                threshold_grid_values.append(threshold_grid_value)

                if include_per_seed:
                    per_seed.append(
                        {
                            "seed": seed,
                            "epsilon": str(eps_attack),
                            "positive_attack_score": positive,
                            "n_positive": 1,
                            "n_negative": len(negatives),
                            "rank_based_auc": rank_auc,
                            "threshold_grid_trapezoidal_value": threshold_grid_value,
                            "threshold_grid_source_field": "roc_by_eps[eps].auc",
                        }
                    )

            row: Dict[str, Any] = {
                "pe": pe,
                "epsilon": str(eps),
                # FINAL13 compact schema consumed by reproduce.py.
                "exact_mean": float(mean(rank_values)),
                "exact_std": _sample_std(rank_values),
                "stored_mean": float(mean(threshold_grid_values)),
                "stored_std": _sample_std(threshold_grid_values),
                "n_seeds": len(seeds),
                # Descriptive aliases/documentation for human readers.
                "exact_estimator": "rank_based_auc",
                "stored_provenance_quantity": "threshold_grid_trapezoidal_value",
                "stored_provenance_field": "roc_by_eps[eps].auc",
                "n_positive_per_seed": 1,
                "n_negative_per_seed": len(_as_float_list(roc[pe][seeds[0]]["all_negative_scores"])),
            }
            if include_per_seed:
                row["per_seed"] = per_seed
            rows.append(row)

    return rows


def print_table(rows: Sequence[Mapping[str, Any]]) -> None:
    print("PE        eps    rank-based AUC     threshold-grid trapezoidal value")
    print("------------------------------------------------------------------")
    for row in rows:
        rank = f"{float(row['exact_mean']):.3f} ± {float(row['exact_std']):.3f}"
        grid = f"{float(row['stored_mean']):.3f} ± {float(row['stored_std']):.3f}"
        print(f"{row['pe']:<9} {str(row['epsilon']):<5} {rank:<18} {grid}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute standard rank-based ROC AUC values from ads_roc_v2.json "
            "and retain archived threshold-grid trapezoidal values as provenance."
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
        help="Path for the derived rank-AUC/provenance JSON. Default: data/ads_roc_rank_auc_sensitivity.json",
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
    parser.add_argument(
        "--include-per-seed",
        action="store_true",
        help="Include per-seed rank-AUC/provenance rows in the output JSON.",
    )
    args = parser.parse_args()

    roc_path = Path(args.roc_path)
    output_path = Path(args.output_path)

    with roc_path.open("r", encoding="utf-8") as f:
        roc = json.load(f)

    rows = compute_rows(
        roc,
        pe_types=args.pe_types,
        epsilons=args.epsilons,
        include_per_seed=args.include_per_seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")

    print_table(rows)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
