import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from processing_data import TopicExtraction
from sentiment_analysis import SentimentWords



def build_data(filenames, min_rev_len=0):
    df_list = []
    for file in filenames:
        df_list.append(pd.read_pickle(file))

    df = pd.concat(df_list)
    df = df.drop_duplicates('reviews')

    df = df[df['review_lengths'] > min_rev_len]

    return df

def build_model(df,
                n_topics,
                ngram_range,
                max_words,
                max_iter,
                top_filename,
                model_filename):
    
    te = TopicExtraction(n_topics=n_topics,
                         sentence=True,
                         ngram_range=ngram_range,
                         max_words=max_words,
                         max_iter=max_iter)

    top_words = te.extract_top_words(df['reviews'],
                                     top_n=15)
                                     # top_filename=top_filename,
                                     # wordcloud=True)

    if model_filename:
        with open(model_filename, 'w') as f:
            dill.dump(te, f)
            print 'Finished pickling the model'
    return te


def main1():
    tic = timeit.default_timer()
    data_SF = '../data/reviews_SF.pkl'
    data_1 = '../data/reviews_1.pkl'
    data_2 = '../data/reviews_2.pkl'

    df =  build_data([data_SF, data_1, data_2], min_rev_len=100)

    toc = timeit.default_timer()
    print 'Data built in %.3f seconds' % (toc - tic)

    n_topics = 100
    ngram_range = (2, 2)
    max_words = 5000
    max_iter = 400

    top_filename = '../data/topics_%d_%dgram_max_%d_100_s.txt' % \
                    (n_topics, ngram_range[1], max_words)

    model_filename = '../data/topics_3.pkl'

    tic = timeit.default_timer()

    te = TopicExtraction(n_topics=n_topics,
                         sentence=True,
                         ngram_range=ngram_range,
                         max_words=max_words,
                         max_iter=max_iter)

    toc = timeit.default_timer()
    print 'Class instanciation in %3.f seconds' % (toc - tic)

    tic = timeit.default_timer()

    top_words = te.extract_top_words(df['reviews'],
                                     top_n=15,
                                     top_filename=top_filename,
                                     wordcloud=False)

    toc = timeit.default_timer()
    print 'Topic extraction in %.3f seconds' % (toc - tic)

    df_SF = pd.read_pickle(data_SF)
    texts = df[df['rest_name'] == '/absinthe-brasserie-and-bar']['reviews']
    tic = timeit.default_timer()
    te.extract_onecat_top(texts, 'wine', 'test_wine')
    toc = timeit.default_timer()
    print 'One category highlights in %3.f seconds' % (toc - tic)
    te.extract_onecat_top(texts, 'experience', 'test_experience')

    te = build_model(df, n_topics, ngram_range, max_words, max_iter, 
                     top_filename, model_filename)


def main2():
    source = '../data/sentiWords/SentiWordNet_3.0.0_20130122.txt'
    sentWord = SentimentWords(source)
    sentWord_filename = '../data/sentWord.pkl'
    with open(sentWord_filename, 'w') as f:
        dill.dump(sentWord, f)
        print 'Finished pickling the model'


if __name__ == '__main__':
    main1()