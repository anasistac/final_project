import pandas as pd
import matplotlib.pyplot as plt

# Load sentiment data
disney_df = pd.read_csv('subtitles_sentiment_roberta_disney.csv')
ghibli_df = pd.read_csv('subtitles_sentiment_roberta_ghibli.csv')

# Calculate sentiment distributions
disney_counts = disney_df['sentiment'].value_counts(normalize=True)
ghibli_counts = ghibli_df['sentiment'].value_counts(normalize=True)

# Combine into a single DataFrame
comparison_df = pd.DataFrame({
    'Disney': disney_counts,
    'Ghibli': ghibli_counts
}).fillna(0)

# Plot the sentiment distributions
plt.figure(figsize=(8, 5))
comparison_df.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Sentiment Distribution Comparison')
plt.ylabel('Proportion of Lines')
plt.xticks(rotation=0)
plt.legend(title='Studio')

# Save the chart to file (e.g., PNG)
plt.savefig("sentiment_comparison_disney_vs_ghibli.png", dpi=300, bbox_inches='tight')

# Optional: show it
plt.show()