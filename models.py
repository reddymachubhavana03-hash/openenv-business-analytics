from pydantic import BaseModel
from typing import List, Dict, Any


class Observation(BaseModel):
    task_id: str
    difficulty: str
    business_question: str
    dataset_schema: Dict[str, Any]
    sample_data: List[Dict[str, Any]]
    available_metrics: List[str]
    constraints: List[str]


class Action(BaseModel):
    kpis: List[str]
    filters: Dict[str, Any]
    aggregation: str
    group_by: List[str]
    reasoning: str
    final_answer: str
    recommendation: str


class Reward(BaseModel):
    score: float
    kpi_score: float = 0.0
    filter_score: float = 0.0
    aggregation_score: float = 0.0
    group_score: float = 0.0
    answer_score: float = 0.0
    recommendation_score: float = 0.0
    feedback: str = ""