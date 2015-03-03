import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, accuracy_score
# from sklearn.pipeline import Pipeline
# import cPickle as pickle

def review_size_distribution(df, filename):
    '''
    INPUT: pandas dataframe
    OUTPUT: none
    Plot the review size distribution for the entire corpus
    '''
    plt.figure(figsize=(10, 8))
    df['review_lengths'].hist(bins=100)
    plt.xlabel('Review length')
    plt.ylabel('Number of reviews')
    plt.title('Review length distribution')
    plt.savefig(filename)

    print 'Review length statistics'
    print df['review_lengths'].describe()

class TopicExtraction(object):
    def __init__(self, n_topics=6, ngram_range=(1, 1), max_words=None):
        self.n_topics = n_topics
        self.max_words = max_words
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(stop_words='english',
                                          max_features=self.max_words,
                                          tokenizer=self.my_tokenize,
                                          ngram_range=self.ngram_range)
        self.factorizor = NMF(n_components=self.n_topics)
        self.H_ = None

    def my_tokenize(self, text):
        '''
        Input: string
        Output: list of strings
        '''
        text = text.lower().encode('ascii', errors='ignore')
        # list_tokenized = RegexpTokenizer(r'\w+').tokenize(text)
        list_tokenized = RegexpTokenizer(r'\W\s+|\s+\W|\W+\b|\b\W+',
                                         gaps=True).tokenize(text)
        list_tokenized = [word for word in list_tokenized if len(word) > 1]
        
        return list_tokenized

    def fit_transform(self, texts):
        V = self.vectorizer.fit_transform(texts)
        W = self.factorizor.fit_transform(V)
        self.H_ = self.factorizor.components_
        
        return W

    def top_words(self, texts, top_n=10):
        self.fit_transform(texts)
        top_words = {}
        for topic in xrange(self.n_topics):
            top_words_idx = np.argsort(self.H_[topic, :])[-1:-(top_n + 1):-1]
            top_words[topic] = [self.vectorizer.get_feature_names()[i] for
                                i in top_words_idx]
        return top_words


if __name__ == '__main__':
    data_SF = '../data/reviews_SF.pkl'
    data_1 = '../data/reviews_1.pkl'
    data_2 = '../data/reviews_2.pkl'
    # length_distribution_filename = '../data/length_distribution.png'

    df_SF = pd.read_pickle(data_SF)
    df1 = pd.read_pickle(data_1)
    df2 = pd.read_pickle(data_2)

    df = pd.concat([df_SF, df1, df2])
    df = df.drop_duplicates('reviews')

    # length_distribution_filename = '../data/length_distribution.png'
    # review_size_distribution(df, length_distribution_filename)
    
    # Getting rid of short reviews (< 100 characters)
    df = df[df['review_lengths'] > 100]   
    
    # te = TopicExtraction(n_topics=20, ngram_range=(1, 1), max_words=1000)
    n_topics = 100
    ngram_range = (1, 1)
    max_words = 5000

    top_filename = '../data/topics_%d_%dgram_max_%d_long.txt' % (n_topics,
                                                                 ngram_range[1],
                                                                 max_words)

    te = TopicExtraction(n_topics=n_topics,
                         ngram_range=ngram_range,
                         max_words=max_words)
    top_words = te.top_words(df['reviews'], top_n=15)

    with open(top_filename, 'w') as f:
        for topic in top_words:
            f.write('Topic %d' % topic)
            f.write('\n')
            for word in top_words[topic]:
                f.write(word)
                f.write('\n')
            f.write('-------------------------')
            f.write('\n')
    f.close()

