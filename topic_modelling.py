# References:
# https://github.com/bloemj/AUC_TM_2025/blob/main/notebooks/12_Clustering_TopicModelling.ipynb

# Imports
import os
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Only run once!
# nltk.download('wordnet') 
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('universal_tagset', download_dir='/home/codespace/nltk_data')
# pip nltk.download('stopwords')

# Data 
data_folder = 'data_preprocessed/disney'

# Define POS tag mapping from Universal → WordNet
un2wn_mapping = {"VERB": wn.VERB, "NOUN": wn.NOUN, "ADJ": wn.ADJ, "ADV": wn.ADV}

# WordNetLemmatizer, POS tags and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Read all files in the data folder
raw_documents = []
movie_titles = []

for filename in os.listdir(data_folder):
    file_path = os.path.join(data_folder, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            words = f.read().split()
            raw_documents.append(words)
            movie_titles.append(filename.replace('.txt', ''))


# Lemmatize with POS awareness
movie_docs = []

for words in raw_documents:
    lemmatized_doc = []
    tagged = nltk.pos_tag(words, tagset="universal")
    for word, tag in tagged:
        if tag in un2wn_mapping:
            lemma = lemmatizer.lemmatize(word, pos=un2wn_mapping[tag])
        else:
            lemma = lemmatizer.lemmatize(word)
        lemma = lemma.lower()
        if tag == "NOUN" and lemma not in stop_words: # only keep nouns
            lemmatized_doc.append(lemma)
    movie_docs.append(lemmatized_doc)

 #''' Construct the document-term matrix '''
# # Imports
from gensim import corpora
import itertools
from collections import Counter
from gensim import models
from gensim.models.ldamodel import LdaModel

# # Filtering out rare and common words based on their frequency in the document 
# # !!! I have randomly chosen 5 and 500 for now --> needs to be changed 
all_words = list(itertools.chain.from_iterable(movie_docs))
word_counts = Counter(all_words)

# Filter each document individually using global word frequencies
filtered_docs = []
for doc in movie_docs:
    filtered = [word for word in doc if 5 <= word_counts[word] <= 500]
    if filtered:
        filtered_docs.append(filtered)

movie_dictionary = corpora.Dictionary(filtered_docs)
movie_bow = [movie_dictionary.doc2bow(doc) for doc in filtered_docs]

print(movie_bow[0][:50])

# # Applying the LDA model
movie_ldamodel = models.ldamodel.LdaModel(movie_bow, num_topics=5, id2word = movie_dictionary, passes = 20)
# Show topics
print("Model topics:")
for topic in movie_ldamodel.show_topics(formatted=False, num_words=5):
    print(topic)

# Get the dominant topic per movie:
print("\nDominant topic per movie:")
for i, bow in enumerate(movie_bow):
    topics = movie_ldamodel.get_document_topics(bow)
    if topics:
        dominant_topic = max(topics, key=lambda x: x[1])
        print(f"{movie_titles[i]}: Dominant topic {dominant_topic[0]}")
    else:
        print(f"Movie {i}: No dominant topic assigned")

# preview the first processed script (just in case)

print("\nExample of processed script:", movie_docs[0][:50])

