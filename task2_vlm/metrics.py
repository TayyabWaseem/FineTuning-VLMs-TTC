"""Pair-aware Task 2 evaluation of already-normalized predictions and references.

This module deliberately requires normalized strings; data preparation and
generation must use one shared VQA normalizer before calling it.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONDITIONS = ("correct", "complementary", "mismatched", "no_image")


def consensus_score(prediction: str, references: list[str]) -> float:
    if len(references) != 10:
        raise ValueError("Natural VQA requires exactly 10 reference answers")
    return sum(min(1.0, sum(prediction == ref for j, ref in enumerate(references) if j != i) / 3)
               for i in range(10)) / 10


def _pair_metrics(rows: list[dict], suite: str) -> dict[str, float]:
    if len(rows) != 2 or len({row["item_id"] for row in rows}) != 2:
        raise ValueError("Each pair must have exactly two distinct examples per condition")
    predictions = [row["prediction_normalized"] for row in rows]
    canonical = [row["canonical_normalized"] for row in rows]
    scores = ([consensus_score(p, row["references_normalized"]) for p, row in zip(predictions, rows)]
              if suite == "natural" else [float(p == answer) for p, answer in zip(predictions, canonical)])
    return {
        "accuracy": float(np.mean(scores)),
        "pair_accuracy": float(all(p == answer for p, answer in zip(predictions, canonical))),
        "identical_answer_rate": float(predictions[0] == predictions[1]),
        "empty_answer_rate": float(np.mean([not p for p in predictions])),
        "reference_nll": float(np.mean([row["reference_nll"] for row in rows])),
        "generation_seconds_per_answer": float(np.mean([row["generation_seconds"] for row in rows])),
        "partner_answer_rate": float(np.mean([p == row.get("partner_canonical_normalized", "<missing>")
                                               for p, row in zip(predictions, rows)])),
    }


def summarize(rows: list[dict], resamples: int = 4000, seed: int = 6304) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["suite"] not in ("natural", "grounding") or row["condition"] not in CONDITIONS:
            raise ValueError("Unknown suite or image condition")
        groups[(row["suite"], row["pair_id"], row["condition"])].append(row)
    suites = sorted({row["suite"] for row in rows})
    output = []
    for suite in suites:
        pair_ids = sorted({row["pair_id"] for row in rows if row["suite"] == suite}, key=str)
        pair_types = {}
        pair_metrics = {}
        for pair_id in pair_ids:
            for condition in CONDITIONS:
                items = groups[(suite, pair_id, condition)]
                if len(items) != 2:
                    raise ValueError(f"Missing or incomplete pair {suite}/{pair_id}/{condition}")
                types = {item["question_type"] for item in items}
                if len(types) != 1:
                    raise ValueError(f"Question type differs within pair {pair_id}")
                pair_types.setdefault(pair_id, next(iter(types)))
                if pair_types[pair_id] != next(iter(types)):
                    raise ValueError(f"Question type differs across conditions for {pair_id}")
                pair_metrics[(pair_id, condition)] = _pair_metrics(items, suite)
            # Each condition must score the same underlying two questions and original answers.
            reference = {row["item_id"]: (row["canonical_normalized"], tuple(row.get("references_normalized", [])))
                         for row in groups[(suite, pair_id, "correct")]}
            for condition in CONDITIONS[1:]:
                other = {row["item_id"]: (row["canonical_normalized"], tuple(row.get("references_normalized", [])))
                         for row in groups[(suite, pair_id, condition)]}
                if reference != other:
                    raise ValueError(f"Original references changed across conditions for {pair_id}")
        for subset in ("all", *sorted(set(pair_types.values()))):
            ids = [p for p in pair_ids if subset == "all" or pair_types[p] == subset]
            rng = np.random.default_rng(seed)
            draws = rng.integers(0, len(ids), size=(resamples, len(ids)))
            for condition in CONDITIONS:
                fields = list(pair_metrics[(ids[0], condition)])
                result = {"suite": suite, "question_type": subset, "condition": condition, "pairs": len(ids)}
                for field in fields:
                    values = np.array([pair_metrics[(p, condition)][field] for p in ids])
                    estimates = values[draws].mean(axis=1)
                    result[field] = float(values.mean())
                    result[field + "_ci_low"], result[field + "_ci_high"] = map(float, np.quantile(estimates, [0.025, 0.975]))
                if condition != "correct":
                    for field in ("accuracy", "reference_nll"):
                        correct = np.array([pair_metrics[(p, "correct")][field] for p in ids])
                        other = np.array([pair_metrics[(p, condition)][field] for p in ids])
                        # Positive NLL difference means the correct image gives lower NLL.
                        difference = other - correct if field == "reference_nll" else correct - other
                        key = "correct_advantage_" + field
                        result[key] = float(difference.mean())
                        result[key + "_ci_low"], result[key + "_ci_high"] = map(float, np.quantile(difference[draws].mean(axis=1), [0.025, 0.975]))
                output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSON array of normalized per-example predictions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    result = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
