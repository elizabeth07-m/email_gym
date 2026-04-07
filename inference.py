"""
Inference script for Message Routing Gym.

Connects to the running OpenEnv server (local or HF Spaces), drives a full
episode with an LLM, and reports the final grade.

Compatible with any OpenAI-compatible API endpoint including:
  - HuggingFace Inference API (fine-tuned Qwen3-0.6B)
  - OpenAI GPT-4o / Turbo
  - Local vLLM / Ollama

Environment variables
---------------------
OPENENV_URL  : URL of the OpenEnv server  [default: http://localhost:8000]
API_BASE_URL : LLM API base URL           [default: https://api.openai.com/v1]
MODEL_NAME   : LLM model identifier       [default: elizabeth07-m/email_gym]
HF_TOKEN     : HuggingFace token (used as API key per hackathon spec)

Quick start
-----------
    # Terminal 1: start the env server
    uvicorn server.app:app --port 8000

    # Terminal 2: run inference
    export OPENENV_URL=http://localhost:8000
    export HF_TOKEN=hf_xxx
    python inference.py
"""

from __future__ import annotations

import os
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
OPENENV_URL  = os.getenv("OPENENV_URL",  "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "elizabeth07-m/email_gym")
HF_TOKEN     = os.getenv("HF_TOKEN",     "dummy-dev-token")

SYSTEM_PROMPT = """\
You are an AI Message Routing Agent operating in an OpenEnv environment.

Your goal is to resolve the `active_directive` by taking one action per step.

Available action types:
  - "route_directory": move a message to a named directory
  - "respond":         dispatch a text reply to the message sender
  - "dismiss":         route the message to vault (archive)

Valid directories: promotions, vault, operations, management

Output ONLY a valid JSON object matching this exact schema:
{
  "action_type": "route_directory" | "respond" | "dismiss",
  "message_id": "<string ID exactly as shown>",
  "target_directory": "<directory name, required if action_type=route_directory>",
  "response_payload": "<reply text, required if action_type=respond>",
  "reasoning": "<1-sentence explanation of this decision>"
}

CRITICAL RULES:
- Output ONLY the raw JSON. No markdown. No backticks.
- Use EXACT message IDs from the queue. Do not invent IDs.
- Read the active_directive carefully — it defines what success looks like.
"""


def build_user_prompt(obs) -> str:
    """Format the current observation as a readable prompt for the LLM."""
    lines = [
        f"ACTIVE DIRECTIVE: {obs.active_directive}",
        f"STEPS REMAINING: {obs.steps_remaining}",
        f"CUMULATIVE REWARD: {obs.cumulative_reward:+.2f}",
        "",
        "MESSAGE QUEUE:",
    ]
    for m in obs.queue:
        level = m.alert_level.value if hasattr(m.alert_level, 'value') else m.alert_level
        lines.append(
            f"  [{m.id}] LEVEL={level.upper()} | FROM={m.source}\n"
            f"       TOPIC: {m.topic}\n"
            f"       CONTENT: {m.content}"
        )

    lines += [
        "",
        f"DIRECTORIES: {json.dumps(obs.directories)}",
    ]
    if obs.step_feedback:
        lines.append(f"LAST FEEDBACK: {obs.step_feedback}")
    if obs.last_execution_error:
        lines.append(f"LAST ERROR: {obs.last_execution_error}")
    if obs.action_history:
        lines.append("ACTION HISTORY: " + " → ".join(obs.action_history[-3:]))

    return "\n".join(lines)


def parse_action(raw: str) -> dict:
    """
    Parse LLM output into a clean action dict.
    Handles common model quirks like markdown fences and trailing commas.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    return json.loads(text)


def main():
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        sys.exit(1)

    try:
        from client import MessageRoutingEnvClient
        from message_routing_gym.models import MessageRoutingAction
    except ImportError as e:
        logger.error(f"Import failed: {e}. Make sure you run from the project root.")
        sys.exit(1)

    logger.info(f"Inference config:")
    logger.info(f"  OPENENV_URL  = {OPENENV_URL}")
    logger.info(f"  API_BASE_URL = {API_BASE_URL}")
    logger.info(f"  MODEL_NAME   = {MODEL_NAME}")

    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    with MessageRoutingEnvClient(base_url=OPENENV_URL) as env:
        logger.info("Checking server health...")
        health = env.health()
        logger.info(f"Server: {health}")

        result = env.reset()
        obs = result.observation

        logger.info(f"\n{'='*60}")
        logger.info(f"Task:      {obs.task_id}")
        logger.info(f"Directive: {obs.active_directive[:100]}...")
        logger.info(f"Queue:     {len(obs.queue)} messages")
        logger.info(f"{'='*60}\n")

        for step in range(1, obs.steps_remaining + 1):
            if not obs.queue and result.done:
                logger.info("Queue empty and episode done.")
                break

            user_prompt = build_user_prompt(obs)

            # ── LLM call ──────────────────────────────────────────
            try:
                completion = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=256,
                )
                raw_output = completion.choices[0].message.content
                action_dict = parse_action(raw_output)
                action = MessageRoutingAction(**action_dict)

                logger.info(
                    f"Step {step:2d} | {action.action_type.upper():18s} "
                    f"#msg={action.message_id}"
                    + (f" → {action.target_directory}" if action.target_directory else "")
                    + (f" | reason: {action.reasoning[:50]}" if action.reasoning else "")
                )

            except Exception as e:
                logger.warning(f"Step {step}: LLM parse error ({e}). Falling back.")
                # Safe fallback: dismiss the first message in queue
                if obs.queue:
                    action = MessageRoutingAction(
                        action_type="dismiss",
                        message_id=obs.queue[0].id,
                    )
                else:
                    logger.info("Nothing to fall back to. Ending episode.")
                    break

            # ── Environment step ──────────────────────────────────
            result = env.step(action)
            obs = result.observation

            logger.info(
                f"         reward={result.reward:+.3f} | "
                f"grade={result.info.get('grade', 0)*100:.1f}% | "
                f"done={result.done}"
            )
            if obs.last_execution_error:
                logger.warning(f"         ERR: {obs.last_execution_error}")

            if result.done:
                final_grade = result.info.get("grade", 0.0)
                passed = final_grade >= 0.80

                logger.info(f"\n{'='*60}")
                logger.info(f"EPISODE COMPLETE")
                logger.info(f"  Final Grade:      {final_grade * 100:.1f}%")
                logger.info(f"  Total Reward:     {obs.cumulative_reward:+.2f}")
                logger.info(f"  Steps Used:       {step}")
                logger.info(f"  Pass (≥80%):      {'✅ YES' if passed else '❌ NO'}")
                logger.info(f"  Curriculum Tier:  {obs.difficulty.upper()}")
                logger.info(f"{'='*60}\n")
                break


if __name__ == "__main__":
    main()
