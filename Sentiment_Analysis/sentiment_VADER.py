# References
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
# Sentiment Analysis of Social Media Text. Eighth International Conference on
# Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.'''

# Imports

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd

nltk.download('punkt_tab')

with open("data_in_sentences/disney_sentences.txt", "r", encoding="utf-8") as file:
    disney_script = file.read()
    
disney_sentences = sent_tokenize(disney_script)
print(disney_sentences [:100])


analyzer = SentimentIntensityAnalyzer()

disney_sentiments = [analyzer.polarity_scores(sentence) for sentence in disney_sentences]
disney_df = pd.DataFrame(disney_sentiments)
disney_df['sentence'] = disney_sentences
print(disney_df.head(30))