import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# --- Configuration ---
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
input_folder = "data_in_sentences/ghibli"  # or ghibli
output_dir = "output_sentiment_curves/ghibli"
n_parts = 10  # Number of parts to divide the movie

# --- Load model & tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
labels = ['negative', 'neutral', 'positive']

# --- Process each file in the folder ---
for filename in os.listdir(input_folder):
    if not filename.endswith(".txt"):
        continue

    file_path = os.path.join(input_folder, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if len(lines) < n_parts:
        continue  # Skip too-short files

    # Split into parts
    chunk_size = len(lines) // n_parts
    parts = [lines[i*chunk_size : (i+1)*chunk_size] for i in range(n_parts)]
    if len(lines) % n_parts != 0:
        parts[-1].extend(lines[n_parts*chunk_size:])

    net_scores = []

    # --- Analyze each part ---
    for part in tqdm(parts, desc=f"Processing {filename}"):
        scores = {'negative': [], 'neutral': [], 'positive': []}
        
        for line in part:
            encoded = tokenizer(line, truncation=True, max_length=514, return_tensors="pt")
            with torch.no_grad():
                output = model(**encoded)
                probs = F.softmax(output.logits, dim=1)[0]
                for i, label in enumerate(labels):
                    scores[label].append(probs[i].item())

        # Average scores for the part
        avg_pos = sum(scores['positive']) / len(scores['positive']) if scores['positive'] else 0
        avg_neg = sum(scores['negative']) / len(scores['negative']) if scores['negative'] else 0
        net_score = avg_pos - avg_neg
        net_scores.append(net_score)

    # --- Plot net sentiment curve ---
    x = list(range(1, n_parts + 1))
    plt.figure(figsize=(10, 4))
    plt.plot(x, net_scores, label="Net Sentiment", color="darkgreen", linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Part of Film")
    plt.ylabel("Net Sentiment (positive - negative)")
    plt.title(filename.replace("_cleaned_sentences.txt", ""))
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, filename.replace(".txt", "_net_sentiment.png"))
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")
