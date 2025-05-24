# Graphs for Topic Modelling Coherence Scores

#Imports
import matplotlib.pyplot as plt

# Coherence scores for Disney
disney_topics = list(range(2, 11))
disney_coherence = [0.2733, 0.2983, 0.3044, 0.3635, 0.3103, 0.3114, 0.3012, 0.3430, 0.3494]

# Coherence scores for Ghibli
ghibli_topics = list(range(2, 11))
ghibli_coherence = [0.2976, 0.3077, 0.3463, 0.3712, 0.3759, 0.3636, 0.4713, 0.4009, 0.4447]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(disney_topics, disney_coherence, marker='o', label='Disney', color='royalblue')
plt.plot(ghibli_topics, ghibli_coherence, marker='o', label='Ghibli', color='forestgreen')
plt.title('LDA Coherence Score Comparison: Disney vs. Ghibli')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score (c_v)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot to file
plt.savefig("disney_vs_ghibli_coherence.png", dpi=300)
print("Plot saved as 'disney_vs_ghibli_coherence.png'")
