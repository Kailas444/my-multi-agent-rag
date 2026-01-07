import gradio as gr
from src.agents.graph import app, AgentState

def run_query(user_query: str, history: list):
    # Safety check for history
    if history is None:
        history = []
        
    if not user_query.strip():
        return history, "", ""

    try:
        inputs = {
            "query": user_query,
            "plan": "",
            "reasoning_trace": [],
            "retrieved_context": "",
            "citations": [],
            "tool_output": "",
            "final_answer": ""
        }

        result = app.invoke(inputs)

        response_text = f"**Answer:**\n{result['final_answer']}\n\n"
        citations_text = ""
        if result.get("citations"):
            citations_text = "**Sources:**\n" + "\n".join([f"- {c['source']} (p. {c['page']})" for c in result["citations"]])
        
        reasoning_text = "**ReAct Trace:**\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(result['reasoning_trace'])])
        history.append((user_query, response_text + citations_text))

        return history, reasoning_text, ""

    except Exception as e:
        # Catch the error and show it in the Trace box
        error_msg = f"**ERROR OCCURRED:**\n{str(e)}"
        return history, error_msg, ""

def create_gradio_app():
    with gr.Blocks(title="Cloud Multi-Agent System") as demo:
        gr.Markdown("# ☁️ Cloud-Deployed Multi-Agent RAG System")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=500, value=[])
                user_input = gr.Textbox(label="Question")
                with gr.Row():
                    submit_btn = gr.Button("Ask")
                    clear_btn = gr.Button("Clear")
            with gr.Column(scale=1):
                trace_box = gr.Textbox(label="ReAct Steps", lines=20, interactive=False)
        
        submit_btn.click(fn=run_query, inputs=[user_input, chatbot], outputs=[chatbot, trace_box, user_input])
        clear_btn.click(fn=lambda: ([], "", ""), inputs=[], outputs=[chatbot, trace_box, user_input])
    
    return demo
