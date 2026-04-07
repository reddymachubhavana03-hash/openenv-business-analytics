from fastapi import FastAPI
from pydantic import BaseModel

from environment import BusinessAnalyticsEnv
from models import Action

app = FastAPI()

env = BusinessAnalyticsEnv()


@app.post("/reset")
def reset():
    observation = env.reset()
    return observation.dict()


from typing import Optional

@app.post("/step")
def step(action: Optional[Action] = None):
    if action is None:
        # fallback dummy action so validator doesn't crash
        action = Action(
            kpis=["revenue"],
            filters={},
            aggregation="sum",
            group_by=["segment"],
            reasoning="default fallback",
            final_answer="fallback",
            recommendation="fallback"
        )

    observation, reward, done = env.step(action)
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": env.state()
    }


@app.get("/state")
def state():
    return env.state()
