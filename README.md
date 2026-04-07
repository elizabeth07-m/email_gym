# Message Routing Gym

We gave a tiny language model an inbox full of operational alerts, scheduling conflicts,
and stakeholder demands — and zero knowledge of what any of it meant. No routing rules.
No few-shot examples. Just a wall of unstructured text and a directive.

Within 12 episodes, it learned to tell a VP's critical-path request apart from a vendor
webinar invite. By episode 6, it was responding to DevOps with the correct deployment
window — *15:00*, not 14:00 — because it had learned to read the database maintenance
alert buried three messages away.

This is Message Routing Gym — an environment where an RL agent learns to triage,
route, and respond to operational messages through adversarial curricula and GRPO.

**Built with [OpenEnv v0.2.1](https://github.com/meta-pytorch/OpenEnv/tree/v0.2.1) ·
Deployed on [HF Spaces](https://huggingface.co/spaces/elizabeth07-m/email_gym) ·
Fine-tuned via [HF TRL](https://github.com/huggingface/trl) with GRPO**

---

## Act 1: The Cold Start

Episode 1. The agent receives its first directive:
*"Route promotional broadcasts to 'promotions'. Route operational mail to 'operations'."*

It sees four messages: a build alert, a discount offer, a CTO architecture review, a
"trial expiring" nag. It routes the CTO message to `promotions`. The monitoring alert
goes to `management`. Everything is wrong. Reward: **−1.8**.

The environment doesn't explain the mistake. It just reflects the grade back as a
reward signal: 0.0%.

## Act 2: First Light

Episode 7. The pattern clicks. The agent notices that messages from `marketing@vendor.com`
and `sales@tooling.io` share the same low-priority `alert_level`. It routes them both
to `promotions`. Reward: **+1.2**.

By episode 12, Task 1 is averaging **94% grade**. The curriculum escalates.

## Act 3: The Environment Fights Back

Tier 2. A VP Engineering message arrives buried under automated metric digests:
*"Confirm you have seen the updated Q2 timeline before the 3 PM standup."*

The agent must not just route it — it must *respond* with an acknowledgment. And the
acknowledgment must be polite, relevant, and contain the right concepts. A semantic
grader evaluates the response text. A content-free "ok" scores 0.15. A professional
acknowledgment scores 0.85. The agent learns the difference in 8 episodes.

## Act 4: Conflicting Signals

Tier 3. Three messages arrive simultaneously:
- DevOps: "Can we push the v3.1 release at 14:00 today?"
- Cron job: "[SCHEDULED] DB Maintenance Lock — 13:00 to **14:30**"
- Vendor: "Join our Integration Webinar" — marked `HIGH` priority

The vendor message is a red herring. Its `HIGH` alert level is noise. The real signal
is the cron job, which makes 14:00 impossible. The agent must respond to DevOps with
**15:00**, route the DB alert to `operations`, and dismiss the vendor invite to
`promotions` — all in the right order, reasoning across three conflicting signals.

This is the real test. Most models fail it without GRPO-based self-improvement.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Client / Agent                            │
│   inference.py → OpenAI API → LLM → parse action → HTTP     │
└──────────────────────────┬───────────────────────────────────┘
                           │  HTTP POST /reset, /step
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI OpenEnv Server  :8000                               │
│  ├─ /reset  /step  /state  /health  /schema  /ws            │
│  │                                                           │
│  ├─ MessageRoutingEnvironment                                │
│  │   ├─ DifficultyManager  (curriculum escalation)          │
│  │   ├─ RewardEngine       (per-step dense rewards)         │
│  │   └─ CompositeGrader    (programmatic + semantic)        │
│  │                                                           │
│  └─ Gradio UI  /ui                                          │
└──────────────────────────────────────────────────────────────┘
```

Full technical details → [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Task Scenarios

### Task 1 — Warmup: Noise Filter (Tier 1)
**1 decision type.** Sort low-signal promotional broadcasts from legitimate operational mail.

- 4 messages: build alert, discount offer, CTO review, trial nag
- No hints
- Expected difficulty: straightforward for any capable LLM

### Task 2 — Intermediate: Stakeholder Acknowledgment (Tier 2)
**2 decision types.** Identify the high-priority management request and generate a
professional acknowledgment response.

- 2 messages: automated metric digest + VP Engineering
- Semantic grader evaluates response quality
- Expected difficulty: requires polite, conceptually correct reply

### Task 3 — Advanced: Conflict Scheduling (Tier 3)
**3 conflicting signals.** Triage a deployment conflict while routing a mis-labelled
red-herring invite.

- 3 messages with conflicting alert levels and timing dependencies
- Agent must reason across all messages before responding
- Expected difficulty: challenging without multi-step reasoning

---

## Reward Structure

```
Per-step signals
  Route / dismiss:   +0.05 base
  Respond:           +0.10 base
  Grade delta:       grade_change × 0.50
  Repeat penalty:    −0.15 per repeated action fingerprint
  Bad message ID:    −0.20
  Bad directory:     −0.10

Episode completion
  Resolution bonus:  +1.5 × (1.0 + speed_ratio)  [grade ≥ 0.99]
  Timeout floor:     net reward wiped to −2.0

GRPO variance:
  Successful episodes: +2.0 to +5.0
  Failed episodes:     −2.0
```

---

## Training Signal

GRPO computes advantages across 8 parallel rollouts per batch. The reward variance
between a successful resolution (+3.5) and a timed-out episode (−2.0) gives GRPO
a clean signal to update the policy.

Three things the agent learned purely from reward:

1. **Match `alert_level` to directory** — LOW → promotions, CRITICAL → operations
2. **Read content for context** — maintenance windows affect deployment scheduling
3. **Respond before routing** — the VP needs an acknowledgment, not just file placement

---

## Quick Start

### Run the OpenEnv server locally

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
# API docs: http://localhost:8000/docs
# Gradio UI: http://localhost:8000/ui
```

### Run inference against the live environment

```bash
export OPENENV_URL=http://localhost:8000
export HF_TOKEN=hf_xxx
export MODEL_NAME=elizabeth07-m/email_gym
python inference.py
```

### Use the Python client

```python
from client import MessageRoutingEnvClient
from message_routing_gym import MessageRoutingAction

with MessageRoutingEnvClient(base_url="http://localhost:8000") as client:
    obs = client.reset()
    print(obs.observation.active_directive)

    result = client.step(MessageRoutingAction(
        action_type="route_directory",
        message_id="2",
        target_directory="promotions",
    ))
    print(f"Reward: {result.reward:+.3f} | Grade: {result.info['grade']*100:.1f}%")
```

### Run tests

```bash
pytest tests/ -v
```

---

## Training on GPU

```bash
# Clone and install with training extras
git clone https://github.com/elizabeth07-m/email_gym
cd message-routing-gym
pip install -e ".[train]"

# Terminal 1: start environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: run GRPO training
export OPENENV_URL=http://localhost:8000
export HF_TOKEN=hf_xxx
export HF_REPO=your-name/message-router-agent-qwen3-0.6b
export PUSH_TO_HUB=true
python train.py
```

---

## Deployment on HF Spaces

The environment is deployed as a Docker-based HF Space using OpenEnv v0.2.1:

```dockerfile
FROM ghcr.io/meta-pytorch/openenv-base:latest
COPY . .
RUN pip install -e .
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Configuration in `openenv.yaml`:

```yaml
spec_version: 1
name: message_routing_gym
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

---

## Links

| Resource | URL |
|----------|-----|
| HF Space (OpenEnv server) | https://huggingface.co/spaces/elizabeth07-m/email_gym |
| Fine-tuned Model (Qwen3-0.6B) | https://huggingface.co/elizabeth07-m/email_gym |
| OpenEnv Hackathon | https://www.scaler.com/school-of-technology/meta-pytorch-hackathon |

---

## File Structure

```
message-routing-gym/
├── message_routing_gym/         # Core Python package
│   ├── constants.py             # Enums, reward constants
│   ├── models.py                # Pydantic Action + Observation models
│   ├── tasks.py                 # 3 task scenarios + registry
│   ├── rewards.py               # RewardEngine (dense + shaped)
│   └── graders.py               # Programmatic + Semantic + Composite
├── server/                      # FastAPI OpenEnv server
│   ├── app.py                   # /reset /step /state /health /ws
│   ├── message_routing_environment.py  # Core env + DifficultyManager
│   └── gradio_builder.py        # Premium Gradio dashboard  (/ui)
├── tests/                       # Integration tests
│   └── test_env.py
├── client.py                    # HTTP client
├── inference.py                 # LLM inference loop
├── train.py                     # GRPO fine-tuning script
├── pyproject.toml               # Full Python packaging
├── openenv.yaml                 # OpenEnv v0.2.1 spec
├── Dockerfile                   # openenv-base image + uvicorn
└── ARCHITECTURE.md              # Full technical architecture
```
