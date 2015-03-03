import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.pipeline import Pipeline
import cPickle as pickle

def review_size_distribution(df, filename):
    '''
    INPUT: pandas dataframe
    OUTPUT: none
    Plot the review size distribution for the entire corpus
    '''
    plt.figure(figsize=(10, 8))
    df['review_lengths'].hist()
    plt.xlabel('Review length')
    plt.ylabel('Number of reviews')
    plt.title('Review length distribution')
    plt.savefig(filename)

    print 'Review length statistics'
    print df['review_lengths'].describe()

class topic_extraction(object):
    def __init__(self, n_topics=6, max_words=1000):
        self.n_topics = n_topics
        self.max_words = max_words
        self.vectorizer = TfidfVectorizer(stop_words='english',
                                          max_features=self.max_words,
                                          tokenizer=my_tokenize)
        self.factorizor = NMF(n_components=self.n_topics)
        self.H_ = None

    def my_tokenize(self, text):
            '''
            Input: string
            Output: list of strings
            '''
            text = text.lower().encode('ascii', errors='ignore')
            list_tokenized = RegexpTokenizer(r'\w+').tokenize(text)
            list_tokenized = text.split()
            list_tokenized = [word for word in list_tokenized if len(word) > 1]
            return list_tokenized

    def fit_transform(texts):
        V = self.vectorizer.fit_transform(texts)
        W = self.factorizor.fit_transform(X)
        H = self.factorizor.components_
        self.H_ = H
        return 



if __name__ == '__main__':
    data_filename = '../data/reviews.pkl'
    length_distribution_filename = '../data/length_distribution.png'
    df = pd.read_pickle(filename)

