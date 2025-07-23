import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import os
# tokenizer and initialisation of the model
#microsoft/Phi-4-mini-instruct
model_id = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

import feedparser

from datetime import datetime

def get_latest_disaster_news(user_input, max_articles=5):

    query = user_input.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"

    feed = feedparser.parse(rss_url)

    if not feed.entries:
        return "No recent news found related to your question."

    # Mots-clés pour filtrer les événements naturels

    keywords = [
    "electrical component",
    "printed circuit board (PCB)",
    "embedded system",
    "sensors",
    "passive components",
    "active components",
    "electric machine",

    "early failure",
    "material fatigue",
    "useful life",
    "residual life",
    "failure modes",
    "aging mechanisms",
    "electrical anomaly",
    "short circuit",
    "overheating",
    "electrical arcing",
    "performance drift",

    "failure prediction",
    "reliability model",
    "predictive maintenance",
    "diagnostics",
    "prognostics",
    "RUL (Remaining Useful Life)",
    "probabilistic estimation",
    "machine learning",
    "predictive models",
    "neural networks",
    "time series analysis",

    "condition monitoring",
    "real-time monitoring",
    "temperature sensors",
    "current sensors",
    "voltage sensors",
    "vibration analysis",
    "thermal analysis",
    "spectral analysis",
    "electrical signals",
    "data acquisition",
    "non-intrusive measurement",

    "failure analysis",

]

    relevant_articles = []
    for entry in feed.entries:
        title = entry.title
        summary = entry.get("summary", "").lower()
        link = entry.link
        date = entry.get("published_parsed")

        # Formatage de la date
        if date:
            pub_date = datetime(*date[:6]).strftime("%Y-%m-%d %H:%M")
        else:
            pub_date = "Date unknown"

        content = f"{title.lower()} {summary}"
        if any(kw in content for kw in keywords):
            relevant_articles.append(f"- [{title}]({link}) _(Published: {pub_date})_")
            if len(relevant_articles) >= max_articles:
                break

    if not relevant_articles:
        return "No relevant disaster-related news found for your question."

    return "Here are the latest disaster-related news articles related to your question:\n\n" + "\n".join(relevant_articles)

# strict system prompt for natural disaster
system_prompt = (
    "You are an AI assistant specialized strictly in electrical fault and machine learning. "
    "Answer only questions about fault in electrical system and machine learning"
    "If the question is unrelated, politely refuse."
)

def chat_with_model(user_input, chat_history=None):
    if chat_history is None:
        chat_history = []

    # 1. Génération initiale du modèle
    messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_input}]
    chat_input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chat_input_tokens = tokenizer(chat_input_text, return_tensors="pt").to(model.device)
    input_length = chat_input_tokens["input_ids"].shape[1]
    max_tokens = min(1024, 4096 - input_length)

    outputs = model.generate(
        **chat_input_tokens,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = outputs[0][input_length:]
    model_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # 2. Détection d'une réponse basée sur des limites de connaissance
    lower_response = model_response.lower()
    triggers = [
        "as of", "my knowledge", "cutoff", "i was trained", "september 2023",
        "i don’t have real-time data", "not up to date", "do not have current", "i'm not updated", "training data ends","I'm sorry,"
    ]
    limited_knowledge = any(trigger in lower_response for trigger in triggers)

    # 3. Recherche seulement si la réponse est affectée par la limite de données
    if limited_knowledge:
        news_summary = get_latest_disaster_news(user_input)
        full_response = (
            f"{model_response}\n\n"
            f"🔎 Since my training data might be outdated, here are some recent news articles related to your question:\n\n"
            f"{news_summary}"
        )
    else:
        full_response = model_response

    # 4. Mise à jour de l’historique
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": full_response})

    return chat_history, chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Electrical fault assistant", type="messages", height=560)
    msg = gr.Textbox(label="Ask questions")
    state = gr.State([])

    with gr.Row():
        send_btn = gr.Button("Send")
        reset_btn = gr.Button("Restart chatbot", variant="stop")

    # Fonction de reset
    def reset_chat():
        initial_state = [{"role": "system", "content": system_prompt}]
        return [], initial_state, ""

    # Envoi message
    send_btn.click(chat_with_model, inputs=[msg, state], outputs=[chatbot, state])
    send_btn.click(lambda: "", None, msg)

    # Reset chatbot
    reset_btn.click(reset_chat, outputs=[chatbot, state, msg])

demo.launch(server_name="0.0.0.0", server_port=7860,share=True)