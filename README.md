---
title: Email Gym
emoji: 📨
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - message-routing
  - llm
  - grpo
  - pytorch
  - meta
pinned: false
short_description: OpenEnv RL env for automated message triage and routing
---

# 📨 Email Gym

An OpenEnv environment where AI agents learn to triage, route, and respond to operational messages through adversarial curricula and GRPO fine-tuning. Built for the Meta × OpenEnv × Hugging Face × PyTorch Hackathon.

## 🎯 Why This Matters

Operational message overload — routing alerts to the wrong team, missing critical VP requests, responding to vendor spam — costs engineering teams hours every week. This environment trains RL agents to be automated message triage specialists, a task humans perform manually every day across DevOps, executive assistants, and operations roles.

Real-world utility: Operations teams, executive assistants, and DevOps engineers manually triage hundreds of messages daily across Slack, email, and ticketing systems. This environment provides a standardised benchmark for training and evaluating agents that automate this process with verifiable, graded outcomes.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Client / Agent                        │
│  inference.py → OpenAI API → LLM → parse action → HTTP  │
└─────────────────────────┬────────────────────────────────┘
                          │ HTTP POST /reset, /step, /state
                          ▼
┌──────────────────────────────────────────────────────────┐
│                Docker Container (HF Space)                │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │              FastAPI (server/app.py)              │    │
│  │   /reset  /step  /state  /health  /schema  /ws   │    │
│  └────────────────────┬─────────────────────────────┘    │
│                       │                                   │
│  ┌────────────────────▼─────────────────────────────┐    │
│  │        MessageRoutingEnvironment                 │    │
│  │        (OpenEnv Environment base class)          │    │
│  │                                                  │    │
│  │  ┌──────────┐ ┌──────────────┐ ┌─────────────┐  │    │
│  │  │  Tasks   │ │ RewardEngine │ │   Graders   │  │    │
│  │  │ Registry │ │ (per-step)   │ │ (0.0→1.0)   │  │    │
│  │  └──────────┘ └──────────────┘ └─────────────┘  │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### Component Diagram

| Component | Responsibility |
|-----------|---------------|
| `message_routing_gym/constants.py` | All enums, config values, reward weights |
| `message_routing_gym/models.py` | Typed Pydantic models (Action, Observation, State) |
| `message_routing_gym/tasks.py` | Task definitions with ground-truth routing rules |
| `message_routing_gym/rewards.py` | Dense reward computation with partial progress |
| `message_routing_gym/graders.py` | Deterministic graders scoring 0.0→1.0 |
| `server/message_routing_environment.py` | OpenEnv Environment with step()/reset()/state() |
| `server/app.py` | FastAPI application wiring + custom Gradio mount |
| `server/gradio_builder.py` | Custom Gradio UI with rich observation display |
| `inference.py` | Baseline agent using OpenAI API |

### Data Flow

```
Agent                     Environment
  │                           │
  ├──── POST /reset ─────────►│  Load task, init RewardEngine
  │◄──── observation ─────────┤  Queue + directive + curriculum tier
  │                           │
  ├──── POST /step ──────────►│  Parse action
  │     {route_directory}     │  Compute reward via RewardEngine
  │◄──── observation ─────────┤  Feedback + reward + done
  │                           │
  ├──── POST /step ──────────►│  Respond action
  │     {respond, payload}    │  Semantic grader evaluates response
  │◄──── observation ─────────┤  Feedback + reward
  │                           │
  ├──── POST /step ──────────►│  Dismiss action
  │     {dismiss}             │  Route to vault, compute grade
  │◄──── observation ─────────┤  Feedback + reward
  │                           │
  ├──── POST /step ──────────►│  Final action
  │                           │  Compute final grader score
  │◄──── observation ─────────┤  done=True + grader_score
  │                           │
```

## 📐 Action & Observation Spaces

### Action Space (MessageRoutingAction)

| Field | Type | UI Widget | Required | Description |
|-------|------|-----------|----------|-------------|
| `action_type` | `"route_directory"` \| `"respond"` \| `"dismiss"` | Dropdown | ✅ | Action to perform |
| `message_id` | string | Textbox | ✅ | Exact ID from the queue |
| `target_directory` | `"promotions"` \| `"operations"` \| `"management"` \| `"vault"` | Dropdown | For route_directory | Destination folder |
| `response_payload` | string | Textarea | For respond | Reply text to dispatch |
| `reasoning` | string | Textarea | Optional | Chain-of-thought explanation |

### Observation Space (MessageRoutingObservation)

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Current task identifier |
| `difficulty` | `"warmup"` \| `"intermediate"` \| `"advanced"` | Curriculum tier |
| `queue` | list[Message] | Messages awaiting triage (id, source, topic, content, alert_level) |
| `directories` | dict[str, int] | Count of messages in each folder |
| `active_directive` | string | Current task goal the agent must resolve |
| `step_feedback` | string | Feedback from last action |
| `steps_remaining` | int | Steps left in episode |
| `cumulative_reward` | float | Running reward total |
| `action_history` | list[str] | Summary of actions taken |
| `last_execution_error` | string | Error from last invalid action |

## 📋 Tasks

### Task 1 — Warmup: Noise Filter (task_warmup_noise)
**1 decision type.** Sort low-signal promotional broadcasts from legitimate operational mail.
- 4 messages: build alert, discount offer, CTO review, trial nag
- Hint provided: `active_directive` explicitly names target directories
- Expected difficulty: Straightforward for any capable LLM

### Task 2 — Intermediate: Stakeholder Acknowledgment (task_intermediate_ack)
**2 decision types.** Identify the high-priority management request and generate a professional acknowledgment response.
- 2 messages: automated metric digest + VP Engineering escalation
- Semantic grader evaluates response quality (polite, conceptually correct)
- Expected difficulty: Requires understanding of urgency and professional tone

### Task 3 — Advanced: Conflict Scheduling (task_advanced_conflict)
**3 conflicting signals.** Triage a deployment conflict while routing a mis-labelled red-herring invite.
- 3 messages: DevOps request, DB maintenance cron alert, vendor invite marked HIGH
- Agent must reason across all messages, respond with correct time (15:00 not 14:00)
- Expected difficulty: Challenging without multi-step reasoning — most models fail without GRPO

## 🎁 Reward Design

Rewards are dense and partial-progress — not binary end-of-episode:

| Action | Correct | Incorrect |
|--------|---------|-----------|
| `route_directory` | +0.05 base + grade delta × 0.50 | −0.10 (bad directory) |
| `respond` | +0.10 base + grade delta × 0.50 | — |
| `dismiss` | +0.05 base + grade delta × 0.50 | — |
| Bad message ID (hallucinated) | — | −0.20 |
| Episode resolution (grade ≥ 0.99) | +1.5 × (1.0 + speed_ratio) | — |
| Timeout floor | — | net reward wiped to −2.0 |

Max score per episode: ~5.0 (fast, perfect resolution)

Grader normalisation: `score = clamp(cumulative_reward / max_reward, 0, 1)`

## 🚀 Setup & Usage

### Prerequisites

- Python 3.10+
- `pip` or `uv`
- Docker (for containerised deployment)

### Environment Variables

```bash
# Copy the example and fill in your secrets
cp .env.example .env

# Edit .env — at minimum set:
#   HF_TOKEN=hf_your_token_here
#   OPENENV_URL=http://localhost:8000
```

### Web Interface (Gradio UI)

When deployed to Hugging Face Spaces (or run locally), the environment provides a custom Gradio web UI at `/ui` with:

- 🔽 Dropdowns for `action_type` and `target_directory`
- 📝 Textbox for `message_id` with queue display
- 📄 Multi-line textarea for `response_payload` and `reasoning`
- 📊 Live metric cards — reward, grade, curriculum tier, step count
- 🖥️ Terminal-style action log with colour-coded rewards
- 📬 Rich message queue cards with alert-level badges

To enable locally:

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
# Then open http://localhost:8000/ui
```

### Local Development

```bash
# Clone the repository
git clone https://github.com/elizabeth07-m/email_gym.git
cd email_gym

# Install dependencies
pip install -e ".[dev]"

# Run the server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v
```

### Docker

```bash
# Build and run
docker compose up --build

# Or manually
docker build -t email-gym .
docker run -p 8000:8000 email-gym
```

### API Usage Examples

```bash
# Health check
curl http://localhost:8000/health

# Reset (warmup task)
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_warmup_noise"}'

# Step (route a message)
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "route_directory", "message_id": "1", "target_directory": "promotions"}}'

# Step (respond to stakeholder)
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "respond", "message_id": "2", "response_payload": "Acknowledged. The deployment window is confirmed for 15:00."}}'

# Step (dismiss to vault)
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "dismiss", "message_id": "3"}}'

# Get state
curl http://localhost:8000/state

# Get schemas
curl http://localhost:8000/schema
```

### Running Inference

```bash
# Export environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your-token-here"
export MODEL_NAME="elizabeth07-m/email_gym"
export OPENENV_URL="http://localhost:8000"

# Run baseline inference
python inference.py
```

## 🚢 Deployment (OpenEnv Push)

This environment is designed for one-command deployment to Hugging Face Spaces via the OpenEnv CLI.

### Step 1 — Validate

```bash
openenv validate
# [OK] email-gym: Ready for multi-mode deployment
```

### Step 2 — Test locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
# Server starts at http://localhost:8000
# Verify: curl http://localhost:8000/health
```

### Step 3 — Deploy to Hugging Face Spaces

```bash
# Login to Hugging Face (if not already)
huggingface-cli login

# Push to your HF Space
openenv push --repo-id elizabeth07-m/email_gym
```

This will:
- Create the `elizabeth07-m/email_gym` Space on Hugging Face (if it doesn't exist)
- Upload all environment files, Dockerfile, and `openenv.yaml`
- Build and deploy the Docker container automatically on HF infrastructure

### Step 4 — Verify deployment

```bash
# Health check (replace with your Space URL)
curl https://elizabeth07-m-email-gym.hf.space/health

# Run inference against the deployed Space
OPENENV_URL="https://elizabeth07-m-email-gym.hf.space" python inference.py
```

### Deployment Options

```bash
# Deploy as a private Space
openenv push --repo-id elizabeth07-m/email_gym --private

# Create a PR instead of pushing directly
openenv push --repo-id elizabeth07-m/email_gym --create-pr
```

## 📊 Baseline Scores

Scores are from the baseline inference agent using `Qwen/Qwen2.5-72B-Instruct`:

| Task | Difficulty | Score | Steps |
|------|------------|-------|-------|
| task_warmup_noise | Warmup | ~0.82 | 4 |
| task_intermediate_ack | Intermediate | ~0.51 | 6 |
| task_advanced_conflict | Advanced | ~0.28 | 8 |
| **Average** | | **~0.54** | |

Scores are approximate and may vary based on model temperature and API availability.

## 📁 Project Structure

```
email-gym/
├── openenv.yaml                               # OpenEnv manifest
├── pyproject.toml                             # Python package config
├── Dockerfile                                 # OpenEnv-compatible build
├── .env.example                               # Environment variable template
├── .gitignore
├── inference.py                               # Baseline inference script
├── client.py                                  # OpenEnv EnvClient wrapper
├── README.md                                  # This file
│
├── message_routing_gym/                       # Core library
│   ├── __init__.py                            # Package exports
│   ├── constants.py                           # Enums, config, reward weights
│   ├── models.py                              # Pydantic Action/Observation/State
│   ├── tasks.py                               # Task definitions + routing rules
│   ├── rewards.py                             # Dense reward engine
│   └── graders.py                             # Deterministic graders (0.0→1.0)
│
├── server/                                    # OpenEnv server
│   ├── __init__.py
│   ├── app.py                                 # FastAPI application
│   ├── gradio_builder.py                      # Custom Gradio web UI
│   └── message_routing_environment.py         # Environment implementation
│
└── tests/                                     # Test suite
    ├── __init__.py
    └── test_env.py                            # Unit + integration tests
```

## 🔗 Links

| Resource | URL |
|----------|-----|
| HF Model / Space | https://huggingface.co/elizabeth07-m/email_gym |
| GitHub Repository | https://github.com/elizabeth07-m/email_gym |
| OpenEnv Hackathon | https://huggingface.co/openenv |
