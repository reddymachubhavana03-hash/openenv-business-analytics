from fastapi import FastAPI
from typing import Optional

from environment import BusinessAnalyticsEnv
from models import Action

app = FastAPI()

env = BusinessAnalyticsEnv()


@app.post("/reset")
def reset():
    observation = env.reset()
    return {
        "task_id": observation.task_id,
        "difficulty": observation.difficulty,
        "business_question": observation.business_question,
        "dataset_schema": observation.dataset_schema,
        "sample_data": observation.sample_data,
        "available_metrics": observation.available_metrics,
        "constraints": observation.constraints,
    }


@app.post("/step")
def step(action: Optional[Action] = None):

    if action is None:
        action = Action(
            kpis=["revenue"],
            filters={},
            aggregation="sum",
            group_by=["segment"],
            reasoning="fallback",
            final_answer="fallback",
            recommendation="fallback"
        )

    observation, reward, done = env.step(action)

    return {
        "observation": {
            "task_id": observation.task_id,
            "difficulty": observation.difficulty,
            "business_question": observation.business_question,
            "dataset_schema": observation.dataset_schema,
            "sample_data": observation.sample_data,
            "available_metrics": observation.available_metrics,
            "constraints": observation.constraints,
        },
        "reward": reward.dict(),
        "done": done,
        "info": env.state(),
    }


@app.get("/state")
def state():
    return env.state()
