from env import BusinessAnalyticsEnv
from models import Action


def rule_based_agent(observation):
    """
    Simple heuristic agent (no API needed)
    """

    return Action(
        kpis=["revenue"],
        filters={},
        aggregation="sum",
        group_by=["region"],
        reasoning="Using revenue KPI grouped by region",
        final_answer="North region has highest revenue",
        recommendation="Focus marketing spend on North region"
    )


if __name__ == "__main__":
    env = BusinessAnalyticsEnv()

    observation = env.reset()
    action = rule_based_agent(observation)

    reward = env.step(action)

    print("\nObservation:")
    print(observation)

    print("\nAction:")
    print(action)

    print("\nReward:")
    print(reward)