import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import re

# Helper: Detect emotional/command phrases
def is_command_or_alert(text):
    return bool(re.match(r"^[A-Z][a-z]*\!$", text.strip())) or text.strip().endswith("!")

# Load new model and tokenizer
model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Label mapping: model predicts 1–5 star ratings
label_map = {
    "LABEL_0": "very negative",
    "LABEL_1": "negative",
    "LABEL_2": "neutral",
    "LABEL_3": "positive",
    "LABEL_4": "very positive"
}

# Set a confidence threshold (optional)
confidence_threshold = 0.45

# Load subtitle lines
with open("data_in_sentences/ghibli/Castle.in.the.Sky_cleaned_sentences.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Process and classify
results = []

for line in tqdm(lines, desc="Classifying lines"):
    encoded = tokenizer(line, truncation=True, max_length=514, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
        probs = F.softmax(output.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        score = probs[0][pred_id].item()
        label_key = f"LABEL_{pred_id}"
        sentiment = label_map[label_key]

        if score < confidence_threshold:
            sentiment = "uncertain"
        elif sentiment == "neutral" and is_command_or_alert(line):
            sentiment = "likely_emotive"

        results.append({
            "text": line,
            "sentiment": sentiment,
            "score": round(score, 4),
            "length": len(line.split())
        })

# Export to CSV
df = pd.DataFrame(results)
df.to_csv("subtitles_sentiment_tabulasiray.csv", index=False)
print(df.head())
