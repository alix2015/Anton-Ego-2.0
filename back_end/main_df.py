import numpy as np 
import pandas as pd 
from pymongo import MongoClient
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from processing_data import TopicExtraction
from sentiment_analysis import SentimentWords, AvgSentimentAnalysis
from sentiment_analysis import BlobSentimentAnalysis
import dill
import timeit
from collections import defaultdict
import random


# Utilities for a pipeline based on dataframes

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
                model_filename,
                top_filename=None):
    
    rest_names = df['rest_name'].dropna().unique().tolist()
    print 'Number of restaurants: %d' % len(rest_names)
    te = TopicExtraction(rest_names=rest_names,
                         n_topics=n_topics,
                         sentence=True,
                         ngram_range=ngram_range,
                         max_words=max_words,
                         max_iter=max_iter)


    if top_filename:
        top_words = te.extract_top_words(df['review'],
                                         top_n=15,
                                         top_filename=top_filename)
                                         # wordcloud=True)
    else:
        te.fit_transform(df['reviews'])

    if model_filename:
        with open(model_filename, 'w') as f:
            dill.dump(te, f)
            print 'Finished pickling the model'
    return te

def main1():
    data_SF = '../../data/reviews_SF.pkl'
    data_1 = '../../data/reviews_1.pkl'
    data_2 = '../../data/reviews_2.pkl'

    df =  build_data([data_SF, data_1, data_2], min_rev_len=100)
    print df.shape

    df.to_pickle('../front_end/data/df.pkl')

    print 'Number of distinct restaurants %d' % (len(df['rest_name'].unique()))

def main2():
    # Using 20 restaurants
    client = MongoClient()
    coll = client.opentable.clean2
    # Fetching the 3500 / 3507 restaurant names that have reviews > 150 char
    cursor = coll.find({'review_length': {'$gt': 150}}, {'rest_name': 1, '_id': 0})
    rest_names = []
    for dic in cursor:
        rest_names.append(dic['rest_name'])
    # Sampling 20 of them
    sample = random.sample(set(rest_names), 20)
    cursor = coll.find({'rest_name': {'$in': sample}})
    df = pd.DataFrame(list(cursor))

    print df.columns
    print df.shape
    print 'Chosen restaurants'
    print df['rest_name'].unique()
    df.to_pickle('../front_end/data/df_clean2.pkl')

def main3():
    df = pd.read_pickle('../front_end/data/df_clean2.pkl')
    texts = df['review']

    n_topics = 100
    ngram_range = (2, 2)
    max_words = 5000
    max_iter = 400

    top_filename = '../../data/te_2_20_%d_%dgram_max_%d_100_s.txt' % \
                    (n_topics, ngram_range[1], max_words)

    model_filename = '../front_end/data/te_2_20.pkl'

    tic = timeit.default_timer()
    te = build_model(df,
                     n_topics,
                     ngram_range,
                     max_words,
                     max_iter,
                     model_filename,
                     top_filename)
    toc = timeit.default_timer()
    print 'Building model in %3.f seconds' % (toc - tic)
    
    # categories = Categories()

    # df_SF = pd.read_pickle(data_SF)
    # texts = df[df['rest_name'] == '/absinthe-brasserie-and-bar']['reviews']
    # tic = timeit.default_timer()
    # te.extract_onecat_topwords(texts, 'wine', 'test5_wine',
    #                            categories=categories)
    # toc = timeit.default_timer()
    # print 'One category highlights in %3.f seconds' % (toc - tic)
    # te.extract_onecat_topwords(texts, 'experience', 'test5_experience')

    # te = build_model(df, n_topics, ngram_range, max_words, max_iter, 
    #                  top_filename, model_filename)

if __name__ == '__main__':
    main3()
