import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm

# Load model and tokenizer
model_name = "sarahai/movie-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Optional: Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label mapping
label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

confidence_threshold = 0.55  # Below this score → label as "uncertain"

# Read the subtitle lines
with open("data_in_sentences/disney/Castle.in.the.Sky_cleaned_sentences.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

results = []

for line in tqdm(lines, desc="Classifying lines"):
    encoded = tokenizer(line, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model(**encoded)
        probs = F.softmax(output.logits, dim=1)
        score = torch.max(probs).item()
        pred_label_id = torch.argmax(probs, dim=1).item()
        raw_label = model.config.id2label[pred_label_id]

    sentiment = label_map.get(raw_label, raw_label)
    if score < confidence_threshold:
        sentiment = "uncertain"

    results.append({
        "text": line,
        "sentiment": sentiment,
        "score": round(score, 4),
        "length": len(line.split())
    })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("subtitles_sentiment_sarahai.csv", index=False)
print(df.head())
