import time
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import os
import pandas as pd
# tokenizer and initialisation of the model
#microsoft/Phi-4-mini-instruct
model_id = "microsoft/Phi-3-mini-4k-instruct"

save = pd.DataFrame(["date","model","question","answer","time","evaluation"])

import feedparser
from datetime import datetime

def get_latest_disaster_news(user_input, max_articles=5):
    """
    Recherche d’actualités sur les catastrophes naturelles en lien avec la question.
    Renvoie un résumé formaté avec titre, lien et date.
    """
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
    "You are a precise and concise AI assistant"
    " Only answer the specific questions asked by the user."
    " Do not generate or invent additional questions or simulate Q&A by yourself."
    " Keep your answers focused, technical, and without self-dialogue."
)

def chat_with_model(user_input, model_id, chat_history=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if chat_history is None:
        chat_history = []

    # SYSTEM + messages
    messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_input}]
    chat_input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chat_input_tokens = tokenizer(chat_input_text, return_tensors="pt").to(model.device)
    input_length = chat_input_tokens["input_ids"].shape[1]
    max_tokens = min(1024, 4096 - input_length)

    # Mesure du temps de réponse
    start_time = time.time()

    outputs = model.generate(
        **chat_input_tokens,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response_time = round(time.time() - start_time, 2)  # en secondes

    generated_tokens = outputs[0][input_length:]
    model_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    lower_response = model_response.lower()
    triggers = [
        "as of", "my knowledge", "cutoff", "i was trained", "september 2023",
        "i don’t have real-time data", "not up to date", "do not have current", "i'm not updated", "training data ends","I'm sorry,"
    ]
    limited_knowledge = any(trigger in lower_response for trigger in triggers)

    if limited_knowledge:
        news_summary = get_latest_disaster_news(user_input)
        full_response = (
            f"{model_response}\n\n"
            f"🔎 Since my training data might be outdated, here are some recent news articles related to your question:\n\n"
            f"{news_summary}"
        )
    else:
        full_response = model_response

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": full_response})

    return chat_history, chat_history, response_time

# Fonction de sauvegarde dans le DataFrame + CSV
def save_to_csv(chat_history, model_id, evaluation, response_time):
    global save
    if not chat_history or len(chat_history) < 2:
        return "⚠️ Nothing to save."

    last_user_msg = chat_history[-2]["content"]
    last_response = chat_history[-1]["content"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame([[timestamp, model_id, last_user_msg, last_response, response_time, evaluation]],
                           columns=["date", "model", "question", "answer", "time", "evaluation"])
    
    save = pd.concat([save, new_row], ignore_index=True)

    file_path = "chat_log.csv"
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, new_row], ignore_index=True)
        combined_df.to_csv(file_path, index=False)
    else:
        new_row.to_csv(file_path, index=False)

    return "✅ Model response saved to chat_log.csv."

# Interface Gradio
with gr.Blocks() as demo:  
    with gr.Row():
        model_id_box_1 = gr.Dropdown(choices=["microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-4-mini-instruct", "meta-llama/Llama-3.1-8B-Instruct", "openai-community/gpt2"], value=model_id, visible=True)
        model_id_box_2 = gr.Dropdown(choices=["microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-4-mini-instruct", "meta-llama/Llama-3.1-8B-Instruct", "openai-community/gpt2"], value=model_id, visible=True)

    with gr.Row():
        chatbot_1 = gr.Chatbot(label="Model 1", type="messages", height=500)
        chatbot_2 = gr.Chatbot(label="Model 2", type="messages", height=500)

    msg = gr.Textbox(label="Ask questions")
    
    state_1 = gr.State([])
    state_2 = gr.State([])

    time_1 = gr.State(0.0)
    time_2 = gr.State(0.0)

    with gr.Row():
        send_btn = gr.Button("Send")
        reset_btn = gr.Button("Reset chatbot", variant="stop")

    def reset_chat():
        return [], [], "", 0.0

    with gr.Row():
        evaluation_1 = gr.Dropdown(label="Evaluate Model 1", choices=["Excellent", "Good", "Average", "Poor", "Irrelevant"])
        save_btn_1 = gr.Button("Save Model 1 Response")

        evaluation_2 = gr.Dropdown(label="Evaluate Model 2", choices=["Excellent", "Good", "Average", "Poor", "Irrelevant"])
        save_btn_2 = gr.Button("Save Model 2 Response")

    # Click behavior
        # Envoi message

    send_btn.click(chat_with_model, inputs=[msg, model_id_box_1, state_1], outputs=[chatbot_1, state_1, time_1])

    send_btn.click(chat_with_model, inputs=[msg, model_id_box_2, state_2], outputs=[chatbot_2, state_2, time_2])

    send_btn.click(lambda: "", None, msg)

    # Reset chatbot
    reset_btn.click(reset_chat, outputs=[chatbot_1, state_1, msg])
    reset_btn.click(reset_chat, outputs=[chatbot_2, state_2, msg])
    save_btn_1.click(fn=save_to_csv, inputs=[state_1, model_id_box_1, evaluation_1, time_1], outputs=[gr.Textbox(visible=False, label="Status Message")])
    save_btn_2.click(fn=save_to_csv, inputs=[state_2, model_id_box_2, evaluation_2, time_2], outputs=[gr.Textbox(visible=False, label="Status Message")])


demo.launch(share=True)

