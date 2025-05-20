import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

labels = ['negative', 'neutral', 'positive']

with open("data_in_sentences/disney.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

results = []

for line in tqdm(lines, desc="Classifying lines"):
    encoded = tokenizer(line, truncation=True, max_length=514, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
        probs = F.softmax(output.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        score = probs[0][pred].item()

    results.append({
    "text": line,
    "sentiment": labels[pred],
    "score": round(score, 4)
    })


df = pd.DataFrame(results)
df.to_csv("subtitles_sentiment_roberta.csv", index=False)
print(df.head())
