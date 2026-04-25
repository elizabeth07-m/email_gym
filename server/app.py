"""
FastAPI OpenEnv server for Message Routing Gym.

Implements the full OpenEnv v0.2.1 HTTP API surface:
  POST /reset       — begin new episode
  POST /step        — advance one step
  GET  /state       — unrestricted internal state (for judges)
  GET  /health      — liveness probe
  GET  /schema      — JSON schemas for action / observation
  WS   /ws          — WebSocket live feed for the Gradio UI

On startup, the Gradio dashboard is also mounted at /ui so the entire
environment is served from a single port (8000).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

from message_routing_gym.models import (
    MessageRoutingAction,
    MessageRoutingObservation,
    ResetRequest,
    StepResponse,
)
from server.message_routing_environment import MessageRoutingEnvironment

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Message Routing Gym",
    description=(
        "An OpenEnv-compliant RL environment where agents learn to triage, "
        "route, and respond to operational messages through adversarial curricula "
        "and GRPO-based self-improvement."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton environment instance.
_env = MessageRoutingEnvironment()

# Connected WebSocket clients (for live Gradio updates)
_ws_clients: list[WebSocket] = []


# ---------------------------------------------------------------------------
# Helper: broadcast state to all WebSocket clients
# ---------------------------------------------------------------------------

async def _broadcast(payload: dict):
    dead: list[WebSocket] = []
    for ws in list(_ws_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness probe — required by OpenEnv / HF Spaces."""
    return {"status": "ok", "environment": "message_routing_gym", "version": "0.1.0"}


@app.post("/reset", response_model=StepResponse)
async def reset(req: ResetRequest = None):
    """
    Start a new episode.

    Selects a task based on the current curriculum tier unless `task_id` is
    provided explicitly (useful for evaluation / grading).
    """
    req = req or ResetRequest()
    result = _env.reset(task_id=req.task_id, seed=req.seed)
    await _broadcast({"event": "reset", "state": _env.state()})
    return result


@app.post("/step", response_model=StepResponse)
async def step(action: MessageRoutingAction):
    """
    Advance the environment by one step.

    The agent sends one `MessageRoutingAction` and receives the next
    `MessageRoutingObservation` along with the step reward and done flag.
    """
    result = _env.step(action)
    await _broadcast({
        "event": "step",
        "state": _env.state(),
        "reward": result.reward,
        "done": result.done,
    })
    return result


@app.get("/state")
async def state():
    """
    Return the full unrestricted environment state.

    Used by external judges, evaluation scripts, and the Gradio UI.
    Not part of the agent-visible observation.
    """
    return _env.state()


@app.get("/schema")
async def schema():
    """Return JSON schemas for action and observation models."""
    return {
        "action_schema": MessageRoutingAction.model_json_schema(),
        "observation_schema": MessageRoutingObservation.model_json_schema(),
    }


# ---------------------------------------------------------------------------
# WebSocket live feed
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for live state streaming to the Gradio UI.
    Clients receive a JSON payload on every reset/step event.
    """
    await ws.accept()
    _ws_clients.append(ws)
    try:
        # Send current state immediately on connect
        await ws.send_json({"event": "connected", "state": _env.state()})
        while True:
            # Keep alive — actual pushes happen via _broadcast()
            await asyncio.sleep(30)
            await ws.send_json({"event": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Mount Gradio UI at /ui
# ---------------------------------------------------------------------------

try:
    from server.gradio_builder import build_gradio_app
    import gradio as gr
    
    gradio_app = build_gradio_app(_env)
    app = gr.mount_gradio_app(app, gradio_app, path="/")
    print("✅ Gradio UI mounted at /")
except ImportError as e:
    print(f"⚠️  Gradio not available, UI skipped: {e}")
except Exception as e:
    print(f"⚠️  Could not mount Gradio UI: {e}")


@app.on_event("startup")
async def startup_event():
    print("🚀 Message Routing Gym server ready.")
    print("   API docs: http://localhost:8000/docs")
    print("   Gradio UI: http://localhost:8000/ui")
