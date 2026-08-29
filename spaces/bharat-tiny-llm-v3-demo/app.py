"""
Bharat-Tiny-LLM v3 - Gradio Chat Demo (for HF Pro Gradio Spaces)
Bilingual Hindi/Hinglish edge LLM, 1.7B params, Q4 quantized, runs offline.

NOTE: HF free Gradio Spaces require a PRO subscription. This file is ready to
deploy on a Pro account, or run locally with:  pip install -r requirements.txt
"""
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import os

MODEL_REPO = "eulogik/Bharat-Tiny-LLM-v3"
MODEL_FILE = "bharat-tiny-llm-v3-q4_k_m.gguf"

SYS = ("You are Bharat-Tiny-LLM, a helpful bilingual assistant for Hindi and "
       "Hinglish. Answer in the same language the user writes. Be concise.")

print("Loading model...", flush=True)
model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=max(2, os.cpu_count() // 2),
    verbose=False,
)
print("Model ready.", flush=True)


def build_prompt(message, history):
    p = "System: " + SYS + "\n"
    for user_msg, bot_msg in history:
        p += "User: " + user_msg + "\n"
        if bot_msg:
            p += "Assistant: " + bot_msg + "\n"
    p += "User: " + message + "\n"
    p += "Assistant:"
    return p


def respond(message, history, temperature, max_tokens):
    prompt = build_prompt(message, history)
    out = llm(
        prompt,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=0.9,
        repeat_penalty=1.15,
        stop=["User:", "Assistant:", "System:"],
        echo=False,
    )
    text = out["choices"][0]["text"].strip()
    history.append((message, text))
    return history, ""


EXAMPLES = [
    "नमस्ते, आपका नाम क्या है?",
    "भारत की राजधानी क्या है?",
    "5 सेब में से 2 खा लिए, बचे कितने?",
    "Hinglish mein batao: aaj ka mausam kaisa hai?",
    "राम के पास 10 रुपये हैं। उसने 4 खर्च किए। बचे कितने?",
]

CSS = """
  .contain { max-width: 760px; margin: auto; }
  footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Bharat-Tiny-LLM v3", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Bharat-Tiny-LLM v3")
    gr.Markdown("**Bilingual Hindi / Hinglish edge LLM** - 1.7B params, runs fully offline. Type in Hindi, Hinglish, or English.")
    chatbot = gr.Chatbot(height=420, label="Chat", bubble_full_width=False)
    with gr.Row():
        msg = gr.Textbox(placeholder="अपना सवाल यहाँ लिखें / Type your question here...", scale=4, container=False)
        send = gr.Button("Send", scale=1, variant="primary")
    with gr.Accordion("Settings", open=False):
        with gr.Row():
            temp = gr.Slider(0.1, 1.0, value=0.4, label="Creativity (temperature)")
            maxtok = gr.Slider(64, 512, value=200, step=32, label="Max length")
    gr.Markdown("**Try:**")
    gr.Examples(EXAMPLES, msg)
    send.click(respond, [msg, chatbot, temp, maxtok], [chatbot, msg])
    msg.submit(respond, [msg, chatbot, temp, maxtok], [chatbot, msg])
    gr.Markdown("---\nModel: `eulogik/Bharat-Tiny-LLM-v3` (Apache 2.0). Runs offline on edge devices via llama.cpp.")

demo.launch()
