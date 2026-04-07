# Architecture — Message Routing Gym

## Overview

Message Routing Gym is an OpenEnv v0.2.1-compliant RL environment where an
LLM agent learns to triage, route, and respond to operational messages. It is
served as a FastAPI application and fine-tuned via GRPO on Qwen3-0.6B.

---

## System Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    Client / Agent                            │
│   inference.py → OpenAI API → LLM → parse action → HTTP     │
└──────────────────────────┬───────────────────────────────────┘
                           │  HTTP POST /reset, /step, /state
                           ▼
┌──────────────────────────────────────────────────────────────┐
│           Docker Container (HF Space)                        │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  FastAPI  server/app.py                                │  │
│  │  /reset  /step  /state  /health  /schema  /ws         │  │
│  └──────────────────────┬─────────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼─────────────────────────────────┐  │
│  │  MessageRoutingEnvironment                             │  │
│  │  (server/message_routing_environment.py)               │  │
│  │                                                        │  │
│  │  ┌────────────────┐  ┌──────────────┐  ┌───────────┐  │  │
│  │  │ DifficultyMgr  │  │ RewardEngine │  │  Tasks    │  │  │
│  │  │ (curriculum)   │  │ (per-step)   │  │ Registry  │  │  │
│  │  └────────────────┘  └──────────────┘  └─────┬─────┘  │  │
│  │                                               │        │  │
│  │  ┌────────────────────────────────────────────▼──────┐ │  │
│  │  │  CompositeGrader                                  │ │  │
│  │  │  ProgrammaticGrader + SemanticGrader → 0.0–1.0   │ │  │
│  │  └───────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Gradio UI  /ui  (server/gradio_builder.py)            │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Component Map

| File | Responsibility |
|------|----------------|
| `message_routing_gym/constants.py` | Enums, reward constants, colour palettes |
| `message_routing_gym/models.py` | Pydantic `MessageRoutingAction` + `MessageRoutingObservation` |
| `message_routing_gym/tasks.py` | 3 task scenarios + `TASK_MAP` registry |
| `message_routing_gym/rewards.py` | `RewardEngine` — per-step reward, repeat penalty, resolution bonus |
| `message_routing_gym/graders.py` | `ProgrammaticGrader`, `SemanticGrader`, `CompositeGrader` |
| `server/message_routing_environment.py` | Core environment: `DifficultyManager` + episode logic |
| `server/app.py` | FastAPI OpenEnv HTTP server; mounts Gradio at `/ui` |
| `server/gradio_builder.py` | Premium dark-mode Gradio dashboard |
| `client.py` | HTTP client (`MessageRoutingEnvClient`) |
| `inference.py` | LLM inference loop (hits HTTP server) |
| `train.py` | GRPO fine-tuning (TRL + live reward signals) |

---

## Data Flow

```
Agent                            Environment
  │                                    │
  ├─── POST /reset ──────────────────►│ Select task (curriculum)
  │◄── observation ───────────────────┤ Active queue + directive
  │                                    │
  ├─── POST /step ───────────────────►│ Parse MessageRoutingAction
  │    (route_directory / respond)    │ Validate message ID + directory
  │                                    │ Update internal state
  │                                    │ RewardEngine.compute_step_reward()
  │                                    │ CompositeGrader.grade()
  │◄── observation + reward ──────────┤ Next obs, step reward, done flag
  │                                    │
  ├─── POST /step ───────────────────►│ ...repeat until done...
  │◄── done=True + grader_score ──────┤ DifficultyManager.update_mastery()
  │                                    │
```

---

## Reward Structure

```
Per-step reward
  ├── Base action reward:    +0.05 (route/dismiss) | +0.10 (respond)
  ├── Grade-delta weight:    grade_delta × 0.50
  ├── Repeat action penalty: −0.15 per repeated fingerprint
  └── Invalid action penalty:−0.20 (bad ID) | −0.10 (bad directory)

Episode completion
  ├── Resolution bonus: +1.5 × (1.0 + speed_ratio)  if grade ≥ 0.99
  └── Timeout floor:   net reward wiped to −2.0       if timeout

GRPO gets clear variance:
  Successful episodes: +2.0 to +5.0
  Failed episodes:     −2.0 (floor)
```

---

## Task Scenarios

| ID | Name | Tier | Grading |
|----|------|------|---------|
| `task_warmup_noise_filter` | Noise Filter | 1 | 100% programmatic |
| `task_intermediate_stakeholder_ack` | Stakeholder Ack | 2 | 50% prog + 50% semantic |
| `task_advanced_conflict_scheduling` | Conflict Scheduling | 3 | 55% prog + 45% semantic |

---

## Curriculum Controller

```
DifficultyManager
  ├── Tracks last N episode grades
  ├── Tier escalation:  avg(last 2) ≥ 0.80 → tier + 1
  └── Tier regression:  avg(last 2) ≤ 0.20 → tier − 1

Tiers:
  1 → Warmup       (task_warmup_noise_filter only)
  2 → Intermediate (Warmup + Intermediate)
  3 → Advanced     (all tasks)
```

---

## Training Loop (GRPO)

```
H100 / A100 GPU                      OpenEnv Server :8000
┌──────────────────────────┐         ┌─────────────────────────┐
│  GRPOTrainer (TRL)       │         │  MessageRoutingEnv      │
│  ├─ Qwen3-0.6B + LoRA    │         │  ├─ DifficultyManager   │
│  ├─ 8 rollouts / batch   │   HTTP  │  ├─ RewardEngine        │
│  └─ GRPO gradient update │◄───────►│  ├─ CompositeGrader     │
│                          │ /reset  │  └─ 3 Task Scenarios    │
│  compute_reward()        │ /step   │                         │
│  └─ live /step calls ───►│         │                         │
└──────────────────────────┘         └─────────────────────────┘
```
