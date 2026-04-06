# OpenEnv Business Analytics Lab

A real-world AI Data Analyst simulation environment built for OpenEnv Hackathon.

## Overview

This environment simulates a business analytics workflow where an AI agent must:

- Choose KPIs
- Apply filters
- Select aggregations
- Provide business recommendations

The environment evaluates decisions using deterministic grading.

## Environment Features

- Real-world business analytics tasks
- Typed Observation, Action, Reward models
- Deterministic scoring (0.0–1.0)
- Reward shaping
- Multiple difficulty levels (easy, medium, hard)
- OpenEnv compliant API (reset, step, state)

## Task Examples

- Revenue KPI selection
- Customer churn analysis
- Segment performance decline

## API

Environment follows OpenEnv interface:

- `reset()` → returns Observation
- `step(Action)` → returns Observation, Reward, Done
- `state()` → internal state

## Files

env.py - main environment
models.py - typed models
grader.py - deterministic scoring
baseline_agent.py - example agent
openenv.yaml - metadata
Dockerfile - deployment
tasks/ - task definitions

## Running Locally

python baseline_agent.py

## Deployment

This environment is Docker-ready and can be deployed to Hugging Face Spaces.

## Author

Hackathon Participant

## License

MIT
