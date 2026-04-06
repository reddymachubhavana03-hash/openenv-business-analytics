import random
from typing import Dict, Any

from models import Observation, Action, Reward
from grader import load_task, grade


class BusinessAnalyticsEnv:
    def __init__(self):
        self.task = None
        self.task_path = None
        self.done = False
        self.step_count = 0
        self.current_observation = None

    def reset(self):
        task_files = [
            "tasks/easy.json",
            "tasks/medium.json",
            "tasks/hard.json"
        ]

        self.task_path = random.choice(task_files)
        self.task = load_task(self.task_path)

        self.done = False
        self.step_count = 0

        self.current_observation = self._get_observation()
        return self.current_observation

    def _get_observation(self):
        return Observation(
            task_id=self.task["task_id"],
            difficulty=self.task["difficulty"],
            business_question=self.task["business_question"],
            dataset_schema=self.task["dataset_schema"],
            sample_data=self.task.get("sample_data", []),
            available_metrics=self.task["available_metrics"],
            constraints=self.task.get("constraints", [])
        )

    def step(self, action: Action):
        if self.done:
            raise ValueError("Episode already completed. Call reset().")

        self.step_count += 1

        reward_dict = grade(action.dict(), self.task)
        reward = Reward(**reward_dict)

        self.done = True

        self.current_observation = self._get_observation()

        info = {
            "task_path": self.task_path,
            "step_count": self.step_count
        }

        return self.current_observation, reward, self.done, info

    def state(self):
        return {
            "task_id": self.task["task_id"] if self.task else None,
            "step_count": self.step_count,
            "done": self.done,
            "observation": self.current_observation.dict() if self.current_observation else None
        }