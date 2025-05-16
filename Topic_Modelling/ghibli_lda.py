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
'''

'''1. Preprocess the data'''

# Imports
import os
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Only run once!
# nltk.download('wordnet') 
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('universal_tagset', download_dir='/home/codespace/nltk_data')
# pip nltk.download('stopwords')

# Data 
data_folder = 'data_preprocessed/ghibli'

# Define POS tag mapping from Universal → WordNet
un2wn_mapping = {"VERB": wn.VERB, "NOUN": wn.NOUN, "ADJ": wn.ADJ, "ADV": wn.ADV}

# WordNetLemmatizer, POS tags and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Read all files in the data folder
raw_documents = []

for filename in os.listdir(data_folder):
    file_path = os.path.join(data_folder, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            words = f.read().split()
            raw_documents.append(words)



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
        if lemma not in stop_words: 
            lemmatized_doc.append(lemma)
    movie_docs.append(lemmatized_doc)

# Training and test sets (80/20)
train_docs, test_docs = train_test_split(movie_docs, test_size=0.3, random_state=42)

print(f"Number of test documents: {len(test_docs)}")

''' 2. Construct the document-term matrix '''

# # Imports
from gensim import corpora
import itertools
from collections import Counter
from gensim import models
from gensim.models.ldamodel import LdaModel

# Create dictionary representation of documents
train_movie_dictionary = corpora.Dictionary(train_docs)

# Filter out words that occur in less than 2 documents, or more than 80% of the documents.
train_movie_dictionary.filter_extremes(no_below=2, no_above=0.8)
print('Number of unique tokens:', len(train_movie_dictionary))

# Bag-of-words representation of the documents
train_bow_corpus = [train_movie_dictionary.doc2bow(d) for d in train_docs]
test_bow_corpus = [train_movie_dictionary.doc2bow(d) for d in test_docs]


''' 3. Applying the LDA model'''
movie_ldamodel = models.ldamodel.LdaModel(train_bow_corpus, num_topics=10, id2word = train_movie_dictionary, passes = 20)

''' 4. Examining the topics'''#

# Top words in each topic (words and weights)
print(movie_ldamodel.show_topics(formatted=False, num_words=5))

# Topic distribution for each document
for doc_topics in movie_ldamodel.get_document_topics(train_bow_corpus):
    print(doc_topics)

''' 5. Evaluation '''
# Imports
from gensim.models import CoherenceModel

# Compute perplexity (how well the model fits the data — lower is better)
print('\nPerplexity:', movie_ldamodel.log_perplexity(test_bow_corpus))

# Compute coherence score (how interpretable the topics are — higher is better)
coherence_model_lda = CoherenceModel(
    model=movie_ldamodel,
    texts=train_docs,
    dictionary=train_movie_dictionary,
    coherence='c_v'  
)

print('\nCoherence score:', coherence_model_lda.get_coherence())


