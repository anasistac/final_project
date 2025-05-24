# References:
# Notebook: https://github.com/bloemj/AUC_TM_2025/blob/main/notebooks/12_Clustering_TopicModelling.ipynb
# ChatGPT: https://chatgpt.com/share/6826fc0b-ea4c-800e-8f39-9801942680cf

'''Topic Modelling with LDA

Steps:
1. Preprocess the data
2. Create a document-term matrix
3. Apply the LDA model
4. Examine the topics
5. Evaluation
6. Visualize the topics
'''

'''1. Preprocess the data'''

# Imports
import os
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from Stopwords_for_lda import stopwords_mallet, ghibli_main_characters, custom_movie_words

# Only run once!
# nltk.download('wordnet') 
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('universal_tagset', download_dir='/home/codespace/nltk_data')
# pip nltk.download('stopwords')

# Data 
data_folder = 'data_split/ghibli'

# Define POS tag mapping from Universal → WordNet
un2wn_mapping = {"VERB": wn.VERB, "NOUN": wn.NOUN, "ADJ": wn.ADJ, "ADV": wn.ADV}

# WordNetLemmatizer, POS tags and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Filtering function
def filter_token(token):
    return token not in stopwords_mallet and token not in ghibli_main_characters and token not in custom_movie_words

raw_documents = []
document_names = []

for filename in sorted(os.listdir(data_folder)):  
    file_path = os.path.join(data_folder, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            words = f.read().split()
            raw_documents.append(words)
            document_names.append(filename)  # track file names for final visulaization


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
        if filter_token(lemma) and tag in {"NOUN"} and len(lemma) > 2: 
            lemmatized_doc.append(lemma)
    movie_docs.append(lemmatized_doc)

print(f"Total documents: {len(movie_docs)}")

''' 2. Construct the document-term matrix '''

# # Imports
from gensim import corpora
import itertools
from collections import Counter
from gensim import models
from gensim.models.ldamodel import LdaModel

# Create dictionary representation of documents
movie_dictionary = corpora.Dictionary(movie_docs)

# Filter out words that occur in less than 2 documents, or more than 80% of the documents.
movie_dictionary.filter_extremes(no_below=3, no_above=0.8)
print('Number of unique tokens:', len(movie_dictionary))

# Bag-of-words representation of the documents
movie_bow_corpus = [movie_dictionary.doc2bow(d) for d in movie_docs]


''' 3. Applying the LDA model'''
movie_ldamodel = models.ldamodel.LdaModel(movie_bow_corpus, num_topics=5, id2word = movie_dictionary, passes = 40)

''' 4. Examining the topics'''#

# Top words in each topic (words and weights)
print(movie_ldamodel.show_topics(formatted=False, num_words=8))

# Topic distribution for each document
for doc_topics in movie_ldamodel.get_document_topics(movie_bow_corpus):
    print(doc_topics)

''' 5. Evaluation '''
# Imports
from gensim.models import CoherenceModel

# Compute coherence score (how interpretable the topics are — higher is better)
coherence_model_lda = CoherenceModel(
    model=movie_ldamodel,
    texts=movie_docs,
    dictionary=movie_dictionary,
    coherence='c_v'  
)

print('\nCoherence score:', coherence_model_lda.get_coherence())

# Compute coherence score for different number of topics
for k in range(2, 21):
    model = LdaModel(movie_bow_corpus, num_topics=k, id2word=movie_dictionary, passes=25)
    coherence = CoherenceModel(model=model, texts=movie_docs, dictionary=movie_dictionary, coherence='c_v').get_coherence()
    print(f"Topics: {k}, Coherence: {coherence:.4f}")

''' 6. Visualize the topics '''
# Imports
import numpy as np
import matplotlib.pyplot as plt

# Create a document-topic matrix: rows = documents, columns = topics
doc2topics = np.zeros((len(movie_docs), movie_ldamodel.num_topics))

# Fill in topic probabilities for each document
for di, doc_topics in enumerate(movie_ldamodel.get_document_topics(movie_bow_corpus, minimum_probability=0)):
    for ti, v in doc_topics:
        doc2topics[di, ti] = v

for i, doc_name in enumerate(document_names):
    print(f"{doc_name}: {doc2topics[i]}")

# Heatmap
docs_id = document_names
num_topics = movie_ldamodel.num_topics

fig = plt.figure(figsize=(16, 12))
plt.pcolor(doc2topics, cmap='Blues')
plt.yticks(np.arange(doc2topics.shape[0]) + 0.5, docs_id, fontsize=10)
plt.xticks(np.arange(num_topics) + 0.5, [f"Topic #{n}" for n in range(num_topics)], rotation=90)
plt.colorbar()
plt.title("Topic Distribution per Document (Heatmap)")
plt.tight_layout()
plt.savefig("ghibli_topic_heatmap.png", dpi=300)
print("Saved heatmap as 'ghibli_topic_heatmap.png'")


# Plot top words per topic

fig = plt.figure(figsize=(16, 10))
num_top_words = 10


topic2top_words = dict(movie_ldamodel.show_topics(formatted=False, num_words=num_top_words))
fontsize_base = 25 / max([w[0][1] for w in topic2top_words.values()])  

for topic, words_shares in topic2top_words.items():
    plt.subplot(1, movie_ldamodel.num_topics, topic + 1)
    plt.ylim(0, num_top_words + 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Topic #{topic}')
    for i, (word, share) in enumerate(words_shares):
        plt.text(0.3, num_top_words - i - 0.5, word, fontsize=fontsize_base * share)

plt.tight_layout()
plt.show()

plt.savefig("ghibli_topic_words_per_topic.png", dpi=300)
print("Saved word-topic plot as 'ghibli_topic_words_per_topic.png'")

