#!/usr/bin/env python3
"""Sweep LSH blocking parameters without running profiling or matching.

This script evaluates only the blocking stage for each configured dataset:
full non-ID attribute embeddings -> LSH candidate generation -> blocking metrics.
It does not call the adaptive profiling prompt or the final pairwise matcher.

Run from the repository root, for example:

    python examples/sweep_blocking_params.py --datasets DBLP-ACM
    python examples/sweep_blocking_params.py --datasets all --tables 5,10,15,20 --planes 4,6,8,10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

loaded_code_module = sys.modules.get("code")
if loaded_code_module is not None and not hasattr(loaded_code_module, "__path__"):
    del sys.modules["code"]

from code.embeddings import embed_dataframe_sbert
from code.loader import load_data
from code.lsh import create_random_planes, query_lsh_fast
from code.utils import build_gt_set, build_id_maps, compute_pc


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    base_path: str
    table_a: str
    table_b: str
    ground_truth: str
    id_a_col: str
    id_b_col: str
    gt_id_a_col: str
    gt_id_b_col: str
    encoding: str = "utf-8"


DATASETS = {
    "DBLP-ACM": DatasetConfig(
        name="DBLP-ACM",
        base_path="datasets/DBLP-ACM",
        table_a="DBLP.csv",
        table_b="ACM.csv",
        ground_truth="gold.csv",
        id_a_col="id",
        id_b_col="id",
        gt_id_a_col="idDBLP",
        gt_id_b_col="idACM",
        encoding="latin1",
    ),
    "Amazon-Walmart": DatasetConfig(
        name="Amazon-Walmart",
        base_path="datasets/Amazon-Walmart",
        table_a="amazon.csv",
        table_b="walmart.csv",
        ground_truth="gold.tsv",
        id_a_col="id",
        id_b_col="id",
        gt_id_a_col="id1",
        gt_id_b_col="id2",
    ),
    "Fodors-Zagat": DatasetConfig(
        name="Fodors-Zagat",
        base_path="datasets/Fodors-Zagat",
        table_a="tableA.csv",
        table_b="tableB.csv",
        ground_truth="gold.csv",
        id_a_col="id",
        id_b_col="id",
        gt_id_a_col="ltable_id",
        gt_id_b_col="rtable_id",
    ),
}


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def selected_datasets(names: str) -> list[DatasetConfig]:
    if names.lower() == "all":
        return list(DATASETS.values())

    configs = []
    for name in [part.strip() for part in names.split(",") if part.strip()]:
        if name not in DATASETS:
            valid = ", ".join(["all", *DATASETS.keys()])
            raise ValueError(f"Unknown dataset {name!r}. Valid choices: {valid}")
        configs.append(DATASETS[name])
    return configs


def load_dataset(config: DatasetConfig):
    base = REPO_ROOT / config.base_path
    df_a = load_data(base / config.table_a, encoding=config.encoding)
    df_b = load_data(base / config.table_b, encoding=config.encoding)
    df_gt = load_data(base / config.ground_truth, encoding=config.encoding)

    id_a_to_pos, id_b_to_pos = build_id_maps(
        df_a,
        df_b,
        config.id_a_col,
        config.id_b_col,
    )
    gt_set = build_gt_set(
        df_gt,
        id_a_to_pos,
        id_b_to_pos,
        config.gt_id_a_col,
        config.gt_id_b_col,
    )

    df_a_block = df_a.drop(columns=[config.id_a_col]).copy()
    df_b_block = df_b.drop(columns=[config.id_b_col]).copy()
    return df_a_block, df_b_block, gt_set


def embed_for_blocking(df_a, df_b, model_name: str, batch_size: int):
    model = SentenceTransformer(model_name)
    table_a_vectors = embed_dataframe_sbert(df_a, model, batch_size=batch_size)
    table_b_vectors = embed_dataframe_sbert(df_b, model, batch_size=batch_size)
    return table_a_vectors, table_b_vectors


def evaluate_lsh_grid(
    config: DatasetConfig,
    model_name: str,
    batch_size: int,
    tables_grid: list[int],
    planes_grid: list[int],
    num_flips: int,
    top_k: int,
    seed: int,
) -> list[dict]:
    print(f"\n=== {config.name} ===")
    df_a, df_b, gt_set = load_dataset(config)
    print(f"Table A shape after ID drop: {df_a.shape}")
    print(f"Table B shape after ID drop: {df_b.shape}")
    print(f"Ground-truth matches: {len(gt_set)}")

    table_a_vectors, table_b_vectors = embed_for_blocking(
        df_a,
        df_b,
        model_name=model_name,
        batch_size=batch_size,
    )
    print(f"Embedding dim: {table_a_vectors.shape[1]}")

    total_pairs = table_a_vectors.shape[0] * table_b_vectors.shape[0]
    rows = []

    for num_tables in tables_grid:
        for num_planes in planes_grid:
            print(
                f"\n[{config.name}] num_tables={num_tables}, "
                f"num_planes={num_planes}, num_flips={num_flips}, top_k={top_k}"
            )
            start = time.time()
            planes_list = create_random_planes(
                num_tables=num_tables,
                num_planes=num_planes,
                dim=table_a_vectors.shape[1],
                seed=seed,
            )
            candidate_pairs = query_lsh_fast(
                table_a_vectors,
                table_b_vectors,
                planes_list,
                num_flips=num_flips,
                top_k=top_k,
            )
            elapsed_seconds = time.time() - start
            pair_completeness, found_matches = compute_pc(candidate_pairs, gt_set)
            candidate_count = len(candidate_pairs)
            reduction_percentage = (
                (total_pairs - candidate_count) / total_pairs * 100
                if total_pairs
                else 0.0
            )

            row = {
                "dataset": config.name,
                "num_tables": num_tables,
                "num_planes": num_planes,
                "num_flips": num_flips,
                "top_k": top_k,
                "candidate_pairs": candidate_count,
                "total_pairs": total_pairs,
                "found_matches": found_matches,
                "gt_matches": len(gt_set),
                "pair_completeness": pair_completeness,
                "reduction_percentage": reduction_percentage,
                "elapsed_seconds": elapsed_seconds,
            }
            rows.append(row)

            print(
                "  candidates={candidate_pairs} | found={found_matches}/{gt_matches} | "
                "PC={pair_completeness:.4f} | reduction={reduction_percentage:.2f}% | "
                "time={elapsed_seconds:.1f}s".format(**row)
            )

    return rows


def print_best_by_dataset(results: pd.DataFrame):
    print("\n=== Best settings by dataset ===")
    for dataset, group in results.groupby("dataset"):
        ranked = group.sort_values(
            by=["pair_completeness", "candidate_pairs", "elapsed_seconds"],
            ascending=[False, True, True],
        )
        best = ranked.iloc[0]
        print(
            f"{dataset}: tables={int(best.num_tables)}, planes={int(best.num_planes)}, "
            f"PC={best.pair_completeness:.4f}, candidates={int(best.candidate_pairs)}, "
            f"reduction={best.reduction_percentage:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Sweep LSH num_tables and num_planes for blocking only."
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset names or 'all'. Choices: DBLP-ACM, Amazon-Walmart, Fodors-Zagat.",
    )
    parser.add_argument("--tables", default="5,10,15,20", help="Comma-separated num_tables values.")
    parser.add_argument("--planes", default="4,6,8,10", help="Comma-separated num_planes values.")
    parser.add_argument("--num-flips", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output",
        default="blocking_sweep_results.csv",
        help="CSV path for sweep results.",
    )
    args = parser.parse_args()

    configs = selected_datasets(args.datasets)
    tables_grid = parse_int_list(args.tables)
    planes_grid = parse_int_list(args.planes)

    all_rows = []
    for config in configs:
        rows = evaluate_lsh_grid(
            config=config,
            model_name=args.model,
            batch_size=args.batch_size,
            tables_grid=tables_grid,
            planes_grid=planes_grid,
            num_flips=args.num_flips,
            top_k=args.top_k,
            seed=args.seed,
        )
        all_rows.extend(rows)

    results = pd.DataFrame(all_rows)
    results = results.sort_values(
        by=["dataset", "pair_completeness", "candidate_pairs", "elapsed_seconds"],
        ascending=[True, False, True, True],
    )
    output_path = REPO_ROOT / args.output
    results.to_csv(output_path, index=False)

    print_best_by_dataset(results)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
