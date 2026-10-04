from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    empty_result = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [],
    }
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()
        metric_names = (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        )
        per_question = [EvalResult(
            question=row["question"],
            answer=row["answer"],
            contexts=row["contexts"],
            ground_truth=row["ground_truth"],
            **{name: float(row.get(name, 0.0)) for name in metric_names},
        ) for _, row in df.iterrows()]
        return {
            **{name: float(df[name].mean()) if name in df else 0.0
               for name in metric_names},
            "per_question": per_question,
        }
    except Exception as error:  # noqa: BLE001 - evaluation must degrade gracefully
        print(f"  ⚠️  RAGAS evaluation failed: {error}")
        return empty_result


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": (
            "Too many irrelevant chunks",
            "Add reranking or metadata filter",
        ),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    failures = []
    for result in eval_results:
        scores = {metric: getattr(result, metric) for metric in diagnostic_tree}
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": sum(scores.values()) / len(scores),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return sorted(failures, key=lambda failure: failure["score"])[:max(bottom_n, 0)]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
