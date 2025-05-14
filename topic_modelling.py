# References:
# https://github.com/bloemj/AUC_TM_2025/blob/main/notebooks/12_Clustering_TopicModelling.ipynb

# Imports
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Only run once!
# nltk.download('wordnet') 
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('universal_tagset', download_dir='/home/codespace/nltk_data')
# nltk.download('stopwords')

# Define POS tag mapping from Universal → WordNet
un2wn_mapping = {"VERB": wn.VERB, "NOUN": wn.NOUN, "ADJ": wn.ADJ, "ADV": wn.ADV}
lemmatizer = WordNetLemmatizer()

# Load the token file (e.g., disney.txt)
with open('data_preprocessed/disney_tokenized.txt', 'r', encoding='utf-8') as f:
    words = f.read().split()  # a long list of tokens

# Get POS tags
tagged = nltk.pos_tag(words, tagset="universal")

# Remove stopwords
stop_words = set(stopwords.words('english'))

# Lemmatize with POS awareness
lemmatized_doc = []
for word, tag in tagged:
    if tag in un2wn_mapping:
        lemma = lemmatizer.lemmatize(word, pos=un2wn_mapping[tag])
    else:
        lemma = lemmatizer.lemmatize(word)
    lemma = lemma.lower()
    if tag == "NOUN" and lemma not in stop_words: # only keep nouns
        lemmatized_doc.append(lemma)

''' Construct the document-term matrix '''
# Imports
from gensim import corpora
import itertools
from collections import Counter
from gensim import models
from gensim.models.ldamodel import LdaModel

# Filtering out rare and common words based on their frequency in the document 
# !!! I have randomly chosen 5 and 500 for now --> needs to be changed 
word_counts = Counter(lemmatized_doc)
filtered_doc = [word for word in lemmatized_doc if 5 <= word_counts[word] <= 500]

# Create pseudo-documents by splitting into chunks, as LDA expects multiple documents
chunk_size = 100  
pseudo_docs = [filtered_doc[i:i + chunk_size] for i in range(0, len(filtered_doc), chunk_size)]

# Remove empty chunks just in case
pseudo_docs = [doc for doc in pseudo_docs if doc]

movie_dictionary = corpora.Dictionary(pseudo_docs)
movie_bow = [movie_dictionary.doc2bow(doc) for doc in pseudo_docs]
print(movie_bow[0][:50])

# Applying the LDA model
movie_ldamodel = models.ldamodel.LdaModel(movie_bow, num_topics=5, id2word = movie_dictionary, passes = 20)
print(movie_ldamodel.show_topics(formatted=False, num_words=5))
