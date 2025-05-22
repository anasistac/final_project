import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm

# Load model and tokenizer
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Define emotion labels from model config
labels = model.config.id2label

# Path to folder with subtitle text files
folder_path = "data_sentences/disney"
output_csv = "disney_emotion_analysis.csv"

results = []

# Iterate over all text files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in tqdm(lines, desc=f"Processing {filename}"):
            encoded = tokenizer(line, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                output = model(**encoded)
                probs = F.softmax(output.logits, dim=1)
                pred_id = torch.argmax(probs, dim=1).item()
                pred_label = labels[pred_id]
                score = probs[0][pred_id].item()

            results.append({
                "text": line,
                "emotion": pred_label,
                "score": round(score, 4),
                "movie": filename.replace("_cleaned_sentences.txt", "")
            })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\n✅ Results saved to: {output_csv}")
