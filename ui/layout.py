import uuid
import gradio as gr
from agent.runner import stream


def reset():
    sid = str(uuid.uuid4())
    return [], "### 🧭 Function Calls", [], [], "", sid


async def on_submit(user_msg, chat, trace, sid):
    async for a, t, chat, trace in stream(user_msg, chat, trace, sid):
        yield a, t, chat, trace


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

        chat_state = gr.State([])
        trace_state = gr.State([])

        gr.Button("🧹 Clear chat history").click(
            reset, None, [chatbox, trace_md, chat_state, trace_state, msg, session_id]
        )

        staged = gr.State("")
        msg.submit(
            lambda x: x, inputs=msg, outputs=staged,
        ).then(
            lambda: "", inputs=None, outputs=msg,
        ).then(
            on_submit, inputs=[staged, chat_state, trace_state, session_id],
            outputs=[chatbox, trace_md, chat_state, trace_state],
        )

        send_btn.click(
            lambda x: x, inputs=msg, outputs=staged,
        ).then(
            lambda: "", inputs=None, outputs=msg,
        ).then(
            on_submit, inputs=[staged, chat_state, trace_state, session_id],
            outputs=[chatbox, trace_md, chat_state, trace_state],
        )

    return demo
