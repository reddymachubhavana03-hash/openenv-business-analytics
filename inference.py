import os
from openai import OpenAI

from environment import BusinessAnalyticsEnv
from models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)


def run_episode(env, task_name="business-analytics"):
    observation = env.reset()
    done = False
    step = 0
    rewards = []

    print(f"[START] task={task_name} env=openenv-business-analytics model={MODEL_NAME}")

    while not done:
        step += 1

        prompt = f"""
You are a business analyst.

Task: {observation.business_question}

Schema: {observation.dataset_schema}
Sample Data: {observation.sample_data}

Return JSON with:
kpis, filters, aggregation, group_by, reasoning, final_answer, recommendation
"""

        error = None

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a business analytics AI."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )

            text = response.choices[0].message.content
            action = Action.parse_raw(text)

        except Exception as e:
            error = str(e)
            action = Action(
                kpis=[],
                filters={},
                aggregation="",
                group_by=[],
                reasoning="",
                final_answer="",
                recommendation=""
            )

        observation, reward, done, _ = env.step(action)

        rewards.append(reward.score)

        print(
            f"[STEP] step={step} action={action} "
            f"reward={reward.score:.2f} done={str(done).lower()} "
            f"error={error if error else 'null'}"
        )

    success = rewards[-1] > 0 if rewards else False
    total_score = sum(rewards) / len(rewards) if rewards else 0

    reward_str = ",".join([f"{r:.2f}" for r in rewards])

    print(
        f"[END] success={str(success).lower()} "
        f"steps={step} score={total_score:.2f} rewards={reward_str}"
    )


def main():
    env = BusinessAnalyticsEnv()
    run_episode(env)


if __name__ == "__main__":
    main()