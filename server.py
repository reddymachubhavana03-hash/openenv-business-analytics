from fastapi import FastAPI
from pydantic import BaseModel

from environment import BusinessAnalyticsEnv
from models import Action

app = FastAPI()

env = BusinessAnalyticsEnv()


@app.post("/reset")
def reset():
    observation = env.reset()
    return observation


@app.post("/step")
def step(action: Action):
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()