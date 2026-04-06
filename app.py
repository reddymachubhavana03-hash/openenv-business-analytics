import gradio as gr
from env import BusinessAnalyticsEnv
from baseline_agent import rule_based_agent


def run_agent():
    env = BusinessAnalyticsEnv()
    observation = env.reset()
    action = rule_based_agent(observation)
    reward = env.step(action)

    return str(observation), str(action), str(reward)


demo = gr.Interface(
    fn=run_agent,
    inputs=[],
    outputs=["text", "text", "text"],
    title="OpenEnv Business Analytics Lab",
    description="Agent interacting with Business Analytics OpenEnv"
)

demo.launch()