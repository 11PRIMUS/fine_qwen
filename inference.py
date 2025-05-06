import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# === Settings ===
base_model_name = "sshleifer/tiny-gpt2"  # change to actual base like Qwen if needed
adapter_path = "emotion"
history_file = "chat_log.json"
MAX_TURNS = 5  # Number of previous exchanges to remember

# === Load tokenizer & model ===
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, adapter_path)

# === Load chat history ===
if os.path.exists(history_file):
    with open(history_file, "r") as f:
        chat_history = json.load(f)
else:
    chat_history = []

def save_chat_history():
    with open(history_file, "w") as f:
        json.dump(chat_history, f, indent=2)

# === Emotion Detection Pipeline ===
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

def detect_emotion(text):
    result = emotion_classifier(text)[0]
    return result['label'].lower()

# === Generate Response ===
def generate_response(user_input, emotion):
    recent_history = chat_history[-MAX_TURNS:]
    history_text = "You are a supportive assistant that adapts responses based on the user's emotions.\n"
    for turn in recent_history:
        history_text += f"User ({turn['emotion']}): {turn['input']}\nAssistant ({turn['emotion']}): {turn['response']}\n"
    history_text += f"User ({emotion}): {user_input}\nAssistant ({emotion}):"

    inputs = tokenizer(history_text, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract the latest assistant reply only
    response = full_output.split(f"Assistant ({emotion}):")[-1].strip().split("User")[0].strip()

    chat_history.append({
        "emotion": emotion,
        "input": user_input,
        "response": response
    })
    save_chat_history()
    return response

# === CLI Interface ===
if __name__ == "__main__":
    print("🧠 Emotion-Aware Chat with Memory\nType 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        emotion = detect_emotion(user_input)
        print(f"Detected Emotion: {emotion}")

        reply = generate_response(user_input, emotion)
        print(f"Assistant ({emotion}): {reply}\n")
