import json
from typing import Dict, Any


def load_task(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def score_kpi(agent_kpis, true_kpis) -> float:
    if not agent_kpis:
        return 0.0

    agent_set = set(agent_kpis)
    true_set = set(true_kpis)

    if agent_set == true_set:
        return 0.25

    overlap = agent_set.intersection(true_set)
    if overlap:
        return 0.12

    return 0.0


def score_filters(agent_filters, true_filters) -> float:
    if not true_filters:
        return 0.15 if not agent_filters else 0.0

    if not agent_filters:
        return 0.0

    agent_keys = set(agent_filters.keys())
    true_keys = set(true_filters.keys())

    if agent_keys == true_keys:
        return 0.15

    overlap = agent_keys.intersection(true_keys)
    if overlap:
        return 0.07

    return 0.0


def score_aggregation(agent_agg, true_agg) -> float:
    if not true_agg:
        return 0.15 if not agent_agg else 0.0

    if agent_agg == true_agg:
        return 0.15

    return 0.0


def score_group_by(agent_group, true_group) -> float:
    if not true_group:
        return 0.10 if not agent_group else 0.0

    if not agent_group:
        return 0.0

    if set(agent_group) == set(true_group):
        return 0.10

    return 0.0


def score_answer(agent_answer, true_answer) -> float:
    if not true_answer:
        return 0.0

    if not agent_answer:
        return 0.0

    agent_answer = agent_answer.lower()
    true_answer = true_answer.lower()

    if true_answer in agent_answer:
        return 0.15

    return 0.0


def score_recommendation(agent_rec, true_rec) -> float:
    if not true_rec:
        return 0.0

    if not agent_rec:
        return 0.0

    agent_rec = agent_rec.lower()
    true_rec = true_rec.lower()

    if any(word in agent_rec for word in true_rec.split()):
        return 0.10

    return 0.0


def grade(agent_action, task):
    truth = task["ground_truth"]

    kpi = score_kpi(
        agent_action.get("kpis"),
        truth.get("kpis")
    )

    filters = score_filters(
        agent_action.get("filters"),
        truth.get("filters")
    )

    aggregation = score_aggregation(
        agent_action.get("aggregation"),
        truth.get("aggregation")
    )

    group = score_group_by(
        agent_action.get("group_by"),
        truth.get("group_by")
    )

    answer = score_answer(
        agent_action.get("final_answer"),
        truth.get("expected_answer")
    )

    recommendation = score_recommendation(
        agent_action.get("recommendation"),
        truth.get("expected_recommendation")
    )

    total = kpi + filters + aggregation + group + answer + recommendation

    return {
        "score": round(min(total, 1.0), 3),
        "kpi_score": kpi,
        "filter_score": filters,
        "aggregation_score": aggregation,
        "group_score": group,
        "answer_score": answer,
        "recommendation_score": recommendation,
        "feedback": "deterministic multi-component grading"
    }