import uuid
import gradio as gr
from collections import deque
from typing import Deque, Dict, Any
from agent.runner import stream
from utils.ui import render_hitl_modal, record_hitl_decision
from utils.agent import render_trace


def reset():
    sid = str(uuid.uuid4())
    return [], "### 🧭 Function Calls", [], [], "", deque(), deque(), sid


async def on_submit(user_msg, chat, trace, sid):
    async for a, t, chat, trace, pending, call_decisions in stream(user_msg, chat, trace, sid, None):
        yield a, t, chat, trace, pending, call_decisions


async def resume_after_hitl(
    chat,
    trace,
    sid: str,
    pending_calls: Deque[Dict[str, Any]] | None,
    call_decisions: Deque[Dict[str, Any]] | None,
):
    """
    Resume the agent graph after human-in-the-loop (HITL) review.

    Behavior:
      - If `pending_calls` is non-empty:
          Do NOT resume the graph. Just push the current chat/trace/pending
          back to the UI so `pending_calls_state.change` can trigger the
          HITL modal.

      - If `pending_calls` is empty and `call_decisions` is non-empty:
          Pass all HITL decisions to `stream` to resume LangGraph execution.

      - If both are empty:
          Do nothing to the graph; just re-emit the current UI state once.
    """
    pending_calls = pending_calls or deque()
    call_decisions = call_decisions or deque()

    if pending_calls:
        yield chat, render_trace(trace), chat, trace, pending_calls, call_decisions
        return

    if not call_decisions:
        yield chat, render_trace(trace), chat, trace, None, None
        return

    async for a, t, chat, trace, pending, call_decisions in stream("", chat, trace, sid, call_decisions):
        yield a, t, chat, trace, pending, call_decisions


def build_ui():
    with gr.Blocks(title="Agent WebUI", css=open("ui/styles.css").read()) as demo:
        session_id = gr.State()
        demo.load(fn=lambda: str(uuid.uuid4()), outputs=session_id)

        gr.Markdown("# ⚙️ Linux Command Agent WebUI")
        with gr.Row():
            with gr.Column(scale=4):
                chatbox = gr.Chatbot(elem_id="chat", type="messages", show_label=False)
            with gr.Column(scale=1, elem_id="trace_col"):
                trace_md = gr.Markdown("### 🧭 Function Calls", elem_id="trace")

        msg = gr.Textbox(placeholder="Enter your question...", show_label=False, lines=2)
        send_btn = gr.Button("🚀 Send", variant="primary")

        with gr.Group(visible=False, elem_id="hitl_modal") as hitl_modal:
            with gr.Column(elem_id="hitl_card"):
                hitl_md = gr.Markdown(
                    "## 🛡️ SafeAgent Runtime Security Control System",
                    elem_id="hitl_modal_content",
                )
                with gr.Row(elem_id="hitl_modal_actions"):
                    approve_btn = gr.Button("APPROVE")
                    shadow_btn = gr.Button("SHADOW RUN")
                    reject_btn = gr.Button("REJECT")

        chat_state = gr.State([])
        trace_state = gr.State([])
        pending_calls_state = gr.State(deque())
        call_decisions_state = gr.State(deque())
        gr.Button("🧹 Clear chat history").click(
            reset,
            None,
            [chatbox, trace_md, chat_state, trace_state, msg, pending_calls_state, call_decisions_state, session_id]
        )

        pending_calls_state.change(
            fn=render_hitl_modal,
            inputs=pending_calls_state,
            outputs=[hitl_md, hitl_modal],
        )

        # APPROVE
        approve_btn.click(
            fn=lambda pending, decisions: record_hitl_decision(pending, decisions, "APPROVE"),
            inputs=[pending_calls_state, call_decisions_state],
            outputs=[pending_calls_state, call_decisions_state],
        ).then(
            fn=render_hitl_modal,
            inputs=pending_calls_state,
            outputs=[hitl_md, hitl_modal],
        ).then(
            fn=resume_after_hitl,
            inputs=[chat_state, trace_state, session_id, pending_calls_state, call_decisions_state],
            outputs=[chatbox, trace_md, chat_state, trace_state, pending_calls_state, call_decisions_state],
        )

        # SHADOW RUN
        shadow_btn.click(
            fn=lambda pending, decisions: record_hitl_decision(pending, decisions, "SHADOW"),
            inputs=[pending_calls_state, call_decisions_state],
            outputs=[pending_calls_state, call_decisions_state],
        ).then(
            fn=render_hitl_modal,
            inputs=pending_calls_state,
            outputs=[hitl_md, hitl_modal],
        ).then(
            fn=resume_after_hitl,
            inputs=[chat_state, trace_state, session_id, pending_calls_state, call_decisions_state],
            outputs=[chatbox, trace_md, chat_state, trace_state, pending_calls_state, call_decisions_state],
        )

        # REJECT
        reject_btn.click(
            fn=lambda pending, decisions: record_hitl_decision(pending, decisions, "REJECT"),
            inputs=[pending_calls_state, call_decisions_state],
            outputs=[pending_calls_state, call_decisions_state],
        ).then(
            fn=render_hitl_modal,
            inputs=pending_calls_state,
            outputs=[hitl_md, hitl_modal],
        ).then(
            fn=resume_after_hitl,
            inputs=[chat_state, trace_state, session_id, pending_calls_state, call_decisions_state],
            outputs=[chatbox, trace_md, chat_state, trace_state, pending_calls_state, call_decisions_state],
        )

        staged = gr.State("")
        msg.submit(
            lambda x: x, inputs=msg, outputs=staged,
        ).then(
            lambda: "", inputs=None, outputs=msg,
        ).then(
            on_submit, inputs=[staged, chat_state, trace_state, session_id],
            outputs=[chatbox, trace_md, chat_state, trace_state, pending_calls_state, call_decisions_state],
        )

        send_btn.click(
            lambda x: x, inputs=msg, outputs=staged,
        ).then(
            lambda: "", inputs=None, outputs=msg,
        ).then(
            on_submit, inputs=[staged, chat_state, trace_state, session_id],
            outputs=[chatbox, trace_md, chat_state, trace_state, pending_calls_state, call_decisions_state],
        )

    return demo
