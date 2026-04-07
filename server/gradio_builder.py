"""
Gradio UI builder for Message Routing Gym.

Provides a premium dark-mode dashboard that visualises the live environment
state via WebSocket polling from the FastAPI server at /state.

Mounted at /ui by server/app.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import gradio as gr
from typing import Any, Optional

from message_routing_gym.constants import DIRECTORY_COLORS, ALERT_COLORS, DirectoryName

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background-color: #080d18 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
.gradio-container { max-width: 1500px !important; margin: 0 auto !important; padding: 16px !important; }

/* ── Header ─────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 4px 40px rgba(99,102,241,0.15);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at top left, rgba(99,102,241,0.12) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 30px; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #818cf8, #38bdf8, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.hero-sub { font-size: 13px; color: #94a3b8; line-height: 1.6; }
.hero-badges { display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
.badge {
    font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 20px;
    background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.4);
    color: #818cf8; letter-spacing: 0.5px;
}
.badge.green { background: rgba(52,211,153,0.1); border-color: rgba(52,211,153,0.4); color: #34d399; }
.badge.blue  { background: rgba(56,189,248,0.1); border-color: rgba(56,189,248,0.4); color: #38bdf8; }

/* ── Metric cards ────────────────────────────────────────── */
.metrics-row { display: flex; gap: 12px; margin-bottom: 20px; }
.metric-card {
    flex: 1; background: rgba(15,23,42,0.8);
    border: 1px solid rgba(51,65,85,0.8); border-radius: 12px;
    padding: 18px; text-align: center;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99,102,241,0.5);
    box-shadow: 0 8px 24px rgba(99,102,241,0.12);
}
.metric-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #64748b; margin-bottom: 8px;
}
.metric-value { font-size: 28px; font-weight: 800; color: #f8fafc; line-height: 1; }
.metric-sub { font-size: 11px; color: #64748b; margin-top: 4px; }

/* ── Panel ───────────────────────────────────────────────── */
.panel {
    background: #0d1526;
    border: 1px solid rgba(51,65,85,0.7);
    border-radius: 14px; padding: 20px; height: 100%;
}
.panel-title {
    font-size: 13px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; color: #94a3b8;
    margin-bottom: 16px; padding-bottom: 12px;
    border-bottom: 1px solid rgba(51,65,85,0.5);
    display: flex; align-items: center; gap: 8px;
}

/* ── Message cards ───────────────────────────────────────── */
.msg-card {
    background: #080d18;
    border-left: 3px solid #3b82f6;
    border-top: 1px solid rgba(51,65,85,0.5);
    border-right: 1px solid rgba(51,65,85,0.5);
    border-bottom: 1px solid rgba(51,65,85,0.5);
    padding: 14px; margin-bottom: 10px; border-radius: 6px;
    transition: all 0.25s ease;
}
.msg-card:hover {
    box-shadow: 0 0 20px rgba(59,130,246,0.18);
    transform: translateX(3px);
}
.msg-id-row { display: flex; justify-content: space-between; margin-bottom: 6px; }
.msg-id { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #64748b; }
.alert-badge {
    font-size: 9px; font-weight: 700; letter-spacing: 1px;
    padding: 2px 8px; border-radius: 20px;
}
.msg-source { font-size: 11px; color: #64748b; margin-bottom: 6px; }
.msg-topic { font-weight: 700; color: #e2e8f0; font-size: 14px; margin-bottom: 5px; }
.msg-content { font-size: 12px; color: #94a3b8; line-height: 1.5; }

/* ── Directive box ───────────────────────────────────────── */
.directive-box {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 8px; padding: 14px; margin-bottom: 16px;
}
.directive-label { font-size: 10px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; color: #64748b; margin-bottom: 6px; }
.directive-text { font-size: 13px; color: #a5b4fc; line-height: 1.6; }

/* ── Directory counters ──────────────────────────────────── */
.dir-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 16px; }
.dir-card {
    background: #080d18; border-radius: 8px; padding: 12px;
    text-align: center; border: 1px solid rgba(51,65,85,0.5);
}
.dir-name { font-size: 9px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 6px; }
.dir-count { font-size: 26px; font-weight: 800; line-height: 1; }

/* ── Terminal ────────────────────────────────────────────── */
.terminal {
    background: #020408;
    border: 1px solid rgba(51,65,85,0.5);
    border-radius: 8px; padding: 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: #22c55e; height: 380px; overflow-y: auto;
    line-height: 1.7;
}
.t-meta { color: #334155; }
.t-act  { color: #38bdf8; }
.t-err  { color: #f87171; }
.t-ok   { color: #4ade80; font-weight: 700; }
.t-warn { color: #facc15; }

/* ── Controls ────────────────────────────────────────────── */
.controls-panel {
    background: #0d1526; border: 1px solid rgba(51,65,85,0.7);
    border-radius: 14px; padding: 20px;
}
"""

# ---------------------------------------------------------------------------
# HTML generators (stateless — receive env reference)
# ---------------------------------------------------------------------------

def _hero_html(env) -> str:
    mgr = env.difficulty_mgr
    task = env._current_task
    task_label = task.task_id if task else "—"
    return f"""
<div class="hero-banner">
  <div class="hero-title">MESSAGE ROUTING GYM</div>
  <div class="hero-sub">
    OpenEnv v0.2.1 · Self-improving RL agent for operational triage ·
    Fine-tuned with GRPO on Qwen3-0.6B
  </div>
  <div class="hero-badges">
    <span class="badge">OpenEnv v0.2.1</span>
    <span class="badge blue">Curriculum: {mgr.level_name.upper()}</span>
    <span class="badge green">FastAPI · Port 8000</span>
    <span class="badge">Task: {task_label}</span>
  </div>
</div>
"""


def _metrics_html(env, last_reward: float, last_grade: float) -> str:
    steps_used = env._step_count
    steps_max = env.max_steps
    cumulative = env._cumulative_reward
    reward_color = "#4ade80" if last_reward > 0 else "#f87171" if last_reward < 0 else "#e2e8f0"
    grade_color = "#4ade80" if last_grade >= 0.8 else "#facc15" if last_grade >= 0.5 else "#f87171"

    return f"""
<div class="metrics-row">
  <div class="metric-card">
    <div class="metric-label">Steps</div>
    <div class="metric-value">{steps_used}<span style="font-size:16px;color:#64748b"> / {steps_max}</span></div>
    <div class="metric-sub">Remaining: {max(0, steps_max - steps_used)}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Last Reward</div>
    <div class="metric-value" style="color:{reward_color}">{last_reward:+.3f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Episode Total</div>
    <div class="metric-value" style="color:#818cf8">{cumulative:+.2f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Grade</div>
    <div class="metric-value" style="color:{grade_color}">{last_grade * 100:.1f}%</div>
  </div>
</div>
"""


def _queue_html(env) -> str:
    obs_queue = env._internal_state.get("queue", [])
    dirs = env._internal_state.get("directories", {})
    dir_counts = {k: len(v) for k, v in dirs.items()}
    task = env._current_task
    directive = task.description if task else "—"

    html = f"""<div class="panel">
  <div class="panel-title">📨 Active Message Queue ({len(obs_queue)} pending)</div>
  <div class="directive-box">
    <div class="directive-label">Directive</div>
    <div class="directive-text">{directive}</div>
  </div>
  <div style="max-height:440px;overflow-y:auto;padding-right:6px;">
"""
    if not obs_queue:
        html += "<div style='text-align:center;padding:40px;color:#64748b;font-style:italic;'>Queue empty — directive complete?</div>"
    else:
        for m in obs_queue:
            level = m.alert_level.value if hasattr(m.alert_level, 'value') else str(m.alert_level)
            border_color = ALERT_COLORS.get(level, "#3b82f6")
            html += f"""
    <div class="msg-card" style="border-left-color:{border_color}">
      <div class="msg-id-row">
        <span class="msg-id">ID: {m.id}</span>
        <span class="alert-badge" style="background:{border_color}22;color:{border_color};border:1px solid {border_color}44">{level.upper()}</span>
      </div>
      <div class="msg-source">📍 {m.source}</div>
      <div class="msg-topic">{m.topic}</div>
      <div class="msg-content">{m.content}</div>
    </div>"""

    html += """  </div>
  <div class="dir-grid">"""
    for dname, dcolor in DIRECTORY_COLORS.items():
        count = dir_counts.get(dname, 0)
        html += f"""
    <div class="dir-card">
      <div class="dir-name" style="color:{dcolor}">{dname}</div>
      <div class="dir-count" style="color:{dcolor if count > 0 else '#334155'}">{count}</div>
    </div>"""
    html += "  </div></div>"
    return html


def _terminal_html(logs: list[str]) -> str:
    html = '<div class="terminal">'
    for line in logs[-40:]:
        html += f"<div>{line}</div>"
    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Gradio app builder
# ---------------------------------------------------------------------------

def build_gradio_app(env: MessageRoutingEnvironment):
    from server.message_routing_environment import MessageRoutingEnvironment
    from message_routing_gym.models import MessageRoutingAction
    from message_routing_gym.constants import ActionType

    action_logs: list[str] = []
    _state = {"reward": 0.0, "grade": 0.0}

    def ts():
        return time.strftime("%H:%M:%S")

    action_logs.append(
        f"<span class='t-meta'>[{ts()}]</span> "
        f"<span class='t-ok'>── SYSTEM INITIALISED. WAITING FOR AGENT... ──</span>"
    )

    # ── Action handler ────────────────────────────────────────────────
    def do_action(action_type, message_id, target_dir, response_payload):
        if env._done:
            action_logs.append(
                f"<span class='t-meta'>[{ts()}]</span> "
                f"<span class='t-warn'>Episode already done. Hit Reset first.</span>"
            )
            return (
                _hero_html(env),
                _metrics_html(env, _state["reward"], _state["grade"]),
                _queue_html(env),
                _terminal_html(action_logs),
            )

        try:
            act = MessageRoutingAction(
                action_type=ActionType(action_type.lower()),
                message_id=str(message_id).strip(),
                target_directory=(target_dir or "").lower().strip(),
                response_payload=(response_payload or "").strip(),
            )
            result = env.step(act)
            _state["reward"] = result.reward
            _state["grade"] = result.info.get("grade", 0.0)

            tag = "t-ok" if result.reward > 0 else "t-err" if result.reward < 0 else "t-act"
            action_logs.append(
                f"<span class='t-meta'>[{ts()}]</span> "
                f"<span class='t-act'>ACT</span> {action_type.upper()} #{message_id}"
                + (f" → {target_dir}" if target_dir else "")
                + f" <span class='{tag}'>r={result.reward:+.3f}</span>"
            )
            if result.observation.last_execution_error:
                action_logs.append(
                    f"<span class='t-meta'>[{ts()}]</span> "
                    f"<span class='t-err'>ERR: {result.observation.last_execution_error}</span>"
                )
            if result.observation.step_feedback:
                action_logs.append(
                    f"<span class='t-meta'>[{ts()}]</span> "
                    f"<span class='t-act'>↳</span> {result.observation.step_feedback}"
                )
            if result.done:
                grade = _state["grade"]
                color = "#4ade80" if grade >= 0.8 else "#f87171"
                action_logs.append(
                    f"<span class='t-meta'>[{ts()}]</span> "
                    f"<span style='color:{color};font-weight:700;'>"
                    f"EPISODE COMPLETE · Grade: {grade*100:.1f}% · "
                    f"Total: {env._cumulative_reward:.2f}</span>"
                )
        except Exception as e:
            action_logs.append(
                f"<span class='t-meta'>[{ts()}]</span> "
                f"<span class='t-err'>SYSTEM ERR: {e}</span>"
            )

        return (
            _hero_html(env),
            _metrics_html(env, _state["reward"], _state["grade"]),
            _queue_html(env),
            _terminal_html(action_logs),
        )

    def do_reset():
        env.reset()
        _state["reward"] = 0.0
        _state["grade"] = 0.0
        action_logs.append(
            f"<span class='t-meta'>[{ts()}]</span> "
            f"<span class='t-ok'>── NEW EPISODE STARTED "
            f"(Curriculum: {env.difficulty_mgr.level_name.upper()}) ──</span>"
        )
        return (
            _hero_html(env),
            _metrics_html(env, 0.0, 0.0),
            _queue_html(env),
            _terminal_html(action_logs),
        )

    # ── Layout ────────────────────────────────────────────────────────
    with gr.Blocks(css=CSS, title="Message Routing Gym") as demo:

        # Header + Metrics
        hero = gr.HTML(_hero_html(env))
        metrics = gr.HTML(_metrics_html(env, 0.0, 0.0))

        with gr.Row():
            # Left: Queue panel
            with gr.Column(scale=5):
                queue_panel = gr.HTML(_queue_html(env))

            # Right: Terminal + Controls
            with gr.Column(scale=4):
                gr.Markdown(
                    "<div class='panel-title'>🖥️ AGENT TERMINAL</div>",
                    elem_classes=["panel"],
                )
                terminal = gr.HTML(_terminal_html(action_logs))

                with gr.Group(elem_classes=["controls-panel"]):
                    gr.Markdown("### ⚡ Manual Action Interface")
                    with gr.Row():
                        act_drop = gr.Dropdown(
                            choices=["route_directory", "respond", "dismiss"],
                            value="route_directory",
                            label="Action Type",
                            scale=1,
                        )
                        msg_id = gr.Textbox(label="Message ID", placeholder="e.g. 1", scale=1)
                    dir_input = gr.Textbox(
                        label="Target Directory",
                        placeholder="promotions | vault | operations | management",
                    )
                    reply_input = gr.Textbox(
                        label="Response Payload",
                        placeholder="Reply text (if action is 'respond')",
                        lines=3,
                    )
                    with gr.Row():
                        exec_btn = gr.Button("▶ Execute Action", variant="primary")
                        reset_btn = gr.Button("↺ Reset Episode", variant="secondary")

        outputs = [hero, metrics, queue_panel, terminal]
        exec_btn.click(do_action, inputs=[act_drop, msg_id, dir_input, reply_input], outputs=outputs)
        reset_btn.click(do_reset, inputs=[], outputs=outputs)

    return demo
