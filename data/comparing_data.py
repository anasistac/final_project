import os
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
from nltk.util import ngrams
from nltk.corpus import stopwords as nltk_stopwords
import nltk

# Download NLTK stopwords (only runs once)
nltk.download('stopwords')

# Paths to preprocessed files
ghibli_path = '/workspaces/final_project/data_preprocessed/ghibli.txt'
disney_path = '/workspaces/final_project/data_preprocessed/disney.txt'

def get_frequencies(filepath):
    """
    Reads a text file, converts it to lowercase, splits it into words,
    and returns the list of words and a Counter of their frequencies.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    words = text.split()
    return words, Counter(words)

def get_bigrams(words):
    """
    Generates bigrams from a list of words and returns a Counter of their frequencies.
    """
    return Counter(ngrams(words, 2))

def get_filtered_counter(original_counter, top_pct=0.0005):
    """
    Creates a new Counter by removing the top percentage of the most common items.
    """
    sorted_items = original_counter.most_common()
    remove_n = int(len(sorted_items) * top_pct)
    # If remove_n is equal to or greater than the length, return an empty Counter to avoid errors
    if remove_n >= len(sorted_items):
        return Counter()
    return Counter(dict(sorted_items[remove_n:]))

def filter_common_uncommon(counter, label, top_pct=0.0005, n=30):
    """
    Prints the N most common items (after filtering top_pct)
    and the N least common items from a Counter.
    """
    sorted_items = counter.most_common()
    remove_n = int(len(sorted_items) * top_pct)
    
    # Ensure filtered_items is not empty
    filtered_items = sorted_items[remove_n:] if remove_n < len(sorted_items) else []

    print(f"\n{label} - {n} most common (after removing top {top_pct*100:.2f}%):")
    for item, freq in filtered_items[:n]:
        print(f"{item}: {freq}")
    
    print(f"\n{label} - {n} least common:")
    # Show least common from the original Counter (without filtering the top 0.05%)
    for item, freq in sorted_items[-n:]:
        print(f"{item}: {freq}")

def show_wordcloud(frequencies, title, ax, colormap='Blues', max_words=100):
    """
    Generates and displays a word cloud from the given frequencies.
    Removes standard stopwords.
    """
    stopwords = set(STOPWORDS) | set(nltk_stopwords.words('english'))
    wc = WordCloud(
        width=800, height=400, background_color='white',
        colormap=colormap, max_words=max_words,
        stopwords=stopwords
    ).generate_from_frequencies(frequencies)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)

def plot_top_words(counter, title, ax, n=20):
    """
    Plots the N most common words from a Counter.
    """
    # Use the Counter directly here; it's expected to be filtered if necessary
    top = counter.most_common(n)
    if not top: # If the counter is empty after filtering
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data to display after filtering', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')
        return

    words, counts = zip(*top)
    ax.bar(words, counts, color='skyblue')
    ax.set_title(title)
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(words, rotation=45, ha='right')
    plt.setp(ax.get_xticklabels(), ha="right", rotation=45) # Improve label rotation

def plot_text_table(ghibli_data, disney_data, title, filename):
    """
    Creates and saves a comparative table of word and bigram frequencies.
    """
    fig, axes = plt.subplots(4, 2, figsize=(20, 20))
    fig.suptitle(title, fontsize=20)

    categories = [
        ('Most Common Words', ghibli_data['common_words'], disney_data['common_words']),
        ('Least Common Words', ghibli_data['rare_words'], disney_data['rare_words']),
        ('Most Common Bigrams', ghibli_data['common_bigrams'], disney_data['common_bigrams']),
        ('Least Common Bigrams', ghibli_data['rare_bigrams'], disney_data['rare_bigrams']),
    ]

    for i, (label, g_data, d_data) in enumerate(categories):
        for j, (data, name) in enumerate(zip([g_data, d_data], ["Ghibli", "Disney"])):
            ax = axes[i, j]
            # Ensure data is in a list of lists format
            table_data = [[k, v] for k, v in data]
            col_labels = ['Text', 'Frequency']
            ax.axis('off')
            ax.set_title(f"{label} - {name}", fontsize=14)
            
            # Adjust table font size to fit well
            table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(10) # Adjust this value as needed
            table.scale(1.2, 1.2) # Adjust table size if necessary

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    
    # Define the percentage to remove from the top common items
    TOP_PERCENT_TO_REMOVE = 0.0005 # This is 0.05%

    # --- Data Processing and Console Output ---
    
    # Store data for the comparative table
    ghibli_data_for_table = {}
    disney_data_for_table = {}

    for label, path, color in [("Ghibli", ghibli_path, 'Blues'), ("Disney", disney_path, 'Reds')]:
        print(f"\n--- {label.upper()} ---")
        
        words, word_counter_original = get_frequencies(path)
        bigram_counter_original = get_bigrams(words)
        
        print(f"Total words: {len(words)}")
        print(f"Unique words: {len(word_counter_original)}")
        
        print("Top 10 words (original):", word_counter_original.most_common(10))
        
        # Print filtered frequencies to the console
        filter_common_uncommon(word_counter_original, f"{label} Words", top_pct=TOP_PERCENT_TO_REMOVE)
        filter_common_uncommon(bigram_counter_original, f"{label} Bigrams", top_pct=TOP_PERCENT_TO_REMOVE)
        
        # Filter Counters for Word Clouds and bar charts
        # Word Clouds and bar charts will now use these filtered frequencies,
        # which DO NOT include the top 0.05% most common words, in addition to standard stopwords.
        word_counter_filtered_for_viz = get_filtered_counter(word_counter_original, top_pct=TOP_PERCENT_TO_REMOVE)
        bigram_counter_filtered_for_viz = get_filtered_counter(bigram_counter_original, top_pct=TOP_PERCENT_TO_REMOVE)


        # Prepare data for the comparative table, ensuring bigrams are also filtered
        def prepare_data_for_table(word_counter, bigram_counter, top_pct):
            sorted_words = word_counter.most_common()
            sorted_bigrams = bigram_counter.most_common()

            remove_n_words = int(len(sorted_words) * top_pct)
            # Ensure there are enough items for the filtered common words list
            common_words = sorted_words[remove_n_words:remove_n_words+30] if remove_n_words < len(sorted_words) else []
            rare_words = sorted_words[-30:] # Least common are always from the original Counter

            remove_n_bigrams = int(len(sorted_bigrams) * top_pct)
            # Ensure there are enough items for the filtered common bigrams list
            common_bigrams = sorted_bigrams[remove_n_bigrams:remove_n_bigrams+30] if remove_n_bigrams < len(sorted_bigrams) else []
            rare_bigrams = sorted_bigrams[-30:] # Least common are always from the original Counter

            format_bigram = lambda b: ' '.join(b[0]) if isinstance(b[0], tuple) else str(b[0])
            
            return {
                'common_words': common_words,
                'rare_words': rare_words,
                'common_bigrams': [(format_bigram(item), freq) for item, freq in common_bigrams],
                'rare_bigrams': [(format_bigram(item), freq) for item, freq in rare_bigrams]
            }

        if label == "Ghibli":
            ghibli_data_for_table = prepare_data_for_table(word_counter_original, bigram_counter_original, TOP_PERCENT_TO_REMOVE)
            ghibli_word_cloud_freq = word_counter_filtered_for_viz # For word clouds and bar plots
        else: # Disney
            disney_data_for_table = prepare_data_for_table(word_counter_original, bigram_counter_original, TOP_PERCENT_TO_REMOVE)
            disney_word_cloud_freq = word_counter_filtered_for_viz # For word clouds and bar plots


    # --- Word Cloud and Bar Chart Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Pass the filtered Counter to show_wordcloud
    show_wordcloud(ghibli_word_cloud_freq, "Most Common Words in Ghibli (Filtered)", axes[0, 0], colormap='Blues')
    show_wordcloud(disney_word_cloud_freq, "Most Common Words in Disney (Filtered)", axes[0, 1], colormap='Reds')
    
    # Pass the filtered Counter to plot_top_words
    plot_top_words(ghibli_word_cloud_freq, 'Top 20 words Ghibli (Filtered)', axes[1, 0])
    plot_top_words(disney_word_cloud_freq, 'Top 20 words Disney (Filtered)', axes[1, 1])

    plt.tight_layout()
    fig.savefig('comparative_wordclouds_and_top_words_filtered.png')
    plt.show()

    # --- Generate Comparison Table ---
    # This uses the data prepared with the 0.05% filter for common words and bigrams
    plot_text_table(ghibli_data_for_table, disney_data_for_table, "Word and Bigram Frequency Comparison (0.05% Filtered)", "text_stats_comparison_filtered.png")

    print("\nProcess completed. Check the generated image files.")