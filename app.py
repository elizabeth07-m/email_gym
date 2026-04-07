import gradio as gr
import json
import time
from env import MessageRoutingEnv
from models import AgentAction, AgentActionType

import logging

env = MessageRoutingEnv()
state = env.reset()
action_logs = []

CSS = """
body {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Inter', system-ui, sans-serif;
}
.gradio-container {
    max-width: 1400px !important;
}
.header-box {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
.title-text {
    font-size: 28px;
    font-weight: 800;
    color: #38bdf8;
    margin-bottom: 8px;
    text-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
}
.subtitle-text {
    font-size: 14px;
    color: #94a3b8;
}
.metric-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    transition: transform 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: #38bdf8;
}
.metric-value {
    font-size: 26px;
    font-weight: bold;
    color: #f8fafc;
}
.metric-label {
    font-size: 12px;
    color: #cbd5e1;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.panel {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    height: 100%;
}
.panel-title {
    font-size: 16px;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.msg-card {
    background: #0f172a;
    border-left: 4px solid #3b82f6;
    border-top: 1px solid #334155;
    border-right: 1px solid #334155;
    border-bottom: 1px solid #334155;
    padding: 12px;
    margin-bottom: 12px;
    border-radius: 4px;
    transition: all 0.2s ease;
}
.msg-card:hover {
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.2);
    border-color: #3b82f6;
}
.msg-header {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 6px;
}
.msg-topic {
    font-weight: bold;
    color: #e2e8f0;
    font-size: 14px;
    margin-bottom: 4px;
}
.msg-content {
    font-size: 13px;
    color: #cbd5e1;
    white-space: pre-wrap;
}
.badge-critical { background: #ef4444; color: white; padding: 2px 6px; border-radius: 10px; font-size: 10px; font-weight: bold; }
.badge-high { background: #f59e0b; color: white; padding: 2px 6px; border-radius: 10px; font-size: 10px; font-weight: bold; }
.badge-normal { background: #3b82f6; color: white; padding: 2px 6px; border-radius: 10px; font-size: 10px; font-weight: bold; }
.badge-low { background: #64748b; color: white; padding: 2px 6px; border-radius: 10px; font-size: 10px; font-weight: bold; }
.terminal-box {
    background: #000;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 12px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    color: #10b981;
    height: 400px;
    overflow-y: auto;
}
.terminal-line { margin-bottom: 4px; }
.terminal-meta { color: #64748b; }
.terminal-action { color: #38bdf8; }
.terminal-error { color: #ef4444; }
"""

def generate_dashboard_html():
    obs = state.observation
    info = getattr(state, 'info', {})
    tier = info.get("curriculum_tier", env.difficulty_mgr.level_name)
    grade = info.get("grade", 0.0)
    
    html = f"""
    <div class="header-box">
        <div class="title-text">MESSAGE_ROUTING :: OPENENV :: AGENT DASHBOARD</div>
        <div class="subtitle-text">Curriculum Tier: <span style="color:#38bdf8; font-weight:bold;">{tier}</span> | Agent: <span style="color:#10b981;">Qwen3-1.5b-GRPO [Idle]</span></div>
    </div>
    
    <div style="display: flex; gap: 16px; margin-bottom: 24px;">
        <div class="metric-card" style="flex: 1;">
            <div class="metric-label">Step Count</div>
            <div class="metric-value">{env.step_count} / {env.max_steps}</div>
        </div>
        <div class="metric-card" style="flex: 1;">
            <div class="metric-label">Immediate Reward</div>
            <div class="metric-value" style="color: {'#10b981' if state.reward > 0 else '#ef4444' if state.reward < 0 else '#e2e8f0'};">
                {state.reward:+.2f}
            </div>
        </div>
        <div class="metric-card" style="flex: 1;">
            <div class="metric-label">Episode Total</div>
            <div class="metric-value" style="color: #38bdf8;">{env.total_episode_reward:+.2f}</div>
        </div>
        <div class="metric-card" style="flex: 1;">
            <div class="metric-label">LLM Integrity Grade</div>
            <div class="metric-value" style="color: #a855f7;">{grade * 100:.1f}%</div>
        </div>
    </div>
    """
    return html

def render_queue_html():
    obs = state.observation
    html = """
    <div class="panel">
        <div class="panel-title">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>
            Active Message Queue
        </div>
        <div style="margin-bottom: 16px; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); padding: 12px; border-radius: 6px;">
            <div style="font-size: 11px; color:#94a3b8; text-transform:uppercase; margin-bottom:4px;">Current Directive</div>
            <div style="font-size: 14px; color:#38bdf8; font-weight: 500;">""" + obs.active_directive + """</div>
        </div>
        <div style="height: 480px; overflow-y: auto; padding-right: 8px;">
    """
    
    if not obs.queue:
        html += "<div style='text-align:center; padding: 40px; color: #64748b; font-style: italic;'>Queue is empty. Directive complete?</div>"
    
    for m in obs.queue:
        bg_color = {
            "critical": "#ef4444", "high": "#f59e0b", "normal": "#3b82f6", "low": "#64748b"
        }.get(m.alert_level.value, "#3b82f6")
        
        html += f"""
        <div class="msg-card" style="border-left-color: {bg_color}">
            <div class="msg-header">
                <span><span style="color:#94a3b8">ID:</span> <span style="color:#e2e8f0; font-family:monospace">{m.id}</span></span>
                <span class="badge-{m.alert_level.value}">{m.alert_level.value.upper()}</span>
            </div>
            <div class="msg-header" style="margin-bottom: 8px;">
                <span>SOURCE: {m.source}</span>
            </div>
            <div class="msg-topic">{m.topic}</div>
            <div class="msg-content">{m.content}</div>
        </div>
        """
        
    html += """
        </div>
        
        <div style="margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap;">
    """
    dir_colors = {"promotions": "#f59e0b", "vault": "#38bdf8", "operations": "#10b981", "management": "#a855f7"}
    for directory, count in obs.directories.items():
        accent = dir_colors.get(directory, "#64748b")
        html += f"""<div style='background: #0f172a; padding: 8px 14px; border-radius: 6px; font-size: 12px; border: 1px solid {accent}55; flex: 1; text-align: center;'>
            <div style='color: {accent}; font-weight: 700; text-transform: uppercase; font-size: 10px; letter-spacing: 1px;'>{directory}</div>
            <div style='color: #f8fafc; font-size: 22px; font-weight: bold; line-height: 1.2;'>{count}</div>
        </div>"""
    
    html += "</div></div>"
    return html

def render_terminal():
    html = '<div class="terminal-box">'
    for log in action_logs[-30:]:
        html += f"<div class='terminal-line'>{log}</div>"
    html += "</div>"
    return html

def process_action(action_type, message_id, target_directory, response_payload):
    global state
    if state.done:
        return generate_dashboard_html(), render_queue_html(), render_terminal(), "Episode Done. Please Reset."
        
    timestamp = time.strftime("%H:%M:%S")
    
    try:
        act_enum = AgentActionType(action_type.lower())
        action = AgentAction(
            action_type=act_enum,
            message_id=message_id.strip(),
            target_directory=target_directory.lower().strip() if target_directory else "",
            response_payload=response_payload.strip() if response_payload else ""
        )
        old_reward = state.reward
        state = env.step(action)
        
        log_entry = f"<span class='terminal-meta'>[{timestamp}]</span> <span class='terminal-action'>ACT:</span> {act_enum.value.upper()} on ID {message_id} "
        if act_enum == AgentActionType.ROUTE_DIRECTORY:
            log_entry += f"-> {target_directory} "
        if act_enum == AgentActionType.RESPOND:
            log_entry += f"(body length {len(response_payload)}) "
            
        reward_delta = state.reward
        log_entry += f"| <span style='color:#a855f7'>Reward: {reward_delta:+.2f}</span>"
        
        if state.observation.last_execution_error:
            error_msg = state.observation.last_execution_error
            log_entry += f"<br><span class='terminal-meta'>[{timestamp}]</span> <span class='terminal-error'>ERR: {error_msg}</span>"
            
        action_logs.append(log_entry)
        
        if state.done:
            info = getattr(state, "info", {})
            grade = info.get("grade", 0.0)
            status_color = "#10b981" if grade > 0.8 else "#ef4444"
            action_logs.append(f"<span class='terminal-meta'>[{timestamp}]</span> <span style='color:{status_color}; font-weight:bold;'>EPISODE COMPLETE! LLM Grade: {grade*100}% | Total R: {env.total_episode_reward:.2f}</span>")
            
        return generate_dashboard_html(), render_queue_html(), render_terminal(), "Success"
        
    except Exception as e:
        err = f"<span class='terminal-meta'>[{timestamp}]</span> <span class='terminal-error'>SYSTEM ERR: {str(e)}</span>"
        action_logs.append(err)
        return generate_dashboard_html(), render_queue_html(), render_terminal(), f"Error: {str(e)}"

def reset_env():
    global state
    state = env.reset()
    timestamp = time.strftime("%H:%M:%S")
    tier_name = getattr(state, "info", {}).get("curriculum_tier", "Warmup")
    action_logs.append(f"<span class='terminal-meta'>[{timestamp}]</span> <span style='color:#38bdf8'>--- NEW EPISODE STARTED (Curriculum: {tier_name}) ---</span>")
    return generate_dashboard_html(), render_queue_html(), render_terminal(), "Reset OK"


with gr.Blocks() as demo:
    dashboard_html_comp = gr.HTML(generate_dashboard_html)
    
    with gr.Row():
        with gr.Column(scale=5):
            queue_html_comp = gr.HTML(render_queue_html)
            
        with gr.Column(scale=4):
            gr.Markdown("<div class='panel-title'>Agent Controls / Logging Terminal</div>")
            terminal_comp = gr.HTML(render_terminal)
            
            with gr.Group():
                gr.Markdown("### Manual Action Overrides")
                with gr.Row():
                    act_dropdown = gr.Dropdown(choices=["dismiss", "route_directory", "respond"], value="route_directory", label="Action Type", scale=1)
                    id_input = gr.Textbox(label="Target Message ID", scale=1)
                folder_input = gr.Textbox(label="Target Directory (if routing)")
                reply_input = gr.Textbox(label="Response Payload (if responding)", lines=2)
                
                with gr.Row():
                    exec_btn = gr.Button("Execute Action via Env", variant="primary")
                    reset_btn = gr.Button("Reset Scenario", variant="secondary")
                    
                status_txt = gr.Textbox(label="System Status", interactive=False)

    exec_btn.click(
        process_action,
        inputs=[act_dropdown, id_input, folder_input, reply_input],
        outputs=[dashboard_html_comp, queue_html_comp, terminal_comp, status_txt]
    )
    
    reset_btn.click(
        reset_env,
        inputs=[],
        outputs=[dashboard_html_comp, queue_html_comp, terminal_comp, status_txt]
    )
    
    # Initialize log
    timestamp = time.strftime("%H:%M:%S")
    action_logs.append(f"<span class='terminal-meta'>[{timestamp}]</span> <span style='color:#38bdf8'>--- SYSTEM INITIALIZED. WAITING FOR AGENT... ---</span>")

if __name__ == "__main__":
    print("Launching Message_RL Dashboard on port 7860...")
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, css=CSS, theme=gr.themes.Monochrome())
