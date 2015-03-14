import numpy as np 
import pandas as pd 
from pymongo import MongoClient
from categories import Categories
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from processing_data import TopicExtraction
# from sentiment_analysis import SentimentWords, AvgSentimentAnalysis
from sentiment_analysis import BlobSentimentAnalysis
import dill
import timeit
from collections import defaultdict
import random


# Utilities for a pipeline based on dataframes

# No longer useful: single pkl file
# def build_data(filenames, min_rev_len=0):
#     df_list = []
#     for file in filenames:
#         df_list.append(pd.read_pickle(file))

#     df = pd.concat(df_list)
#     df = df.drop_duplicates('reviews')

#     df = df[df['review_lengths'] > min_rev_len]

#     return df

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

# def main1():
#     data_SF = '../../data/reviews_SF.pkl'
#     data_1 = '../../data/reviews_1.pkl'
#     data_2 = '../../data/reviews_2.pkl'

#     df =  build_data([data_SF, data_1, data_2], min_rev_len=100)
#     print df.shape

#     df.to_pickle('../front_end/data/df.pkl')

#     print 'Number of distinct restaurants %d' % (len(df['rest_name'].unique()))

# Random sampling not user-friendly
# def main2():
#     # Using 20 restaurants
#     client = MongoClient()
#     coll = client.opentable.clean2
#     # Fetching the 3500 / 3507 restaurant names that have reviews > 150 char
#     cursor = coll.find({'review_length': {'$gt': 150}}, {'rest_name': 1, '_id': 0})
#     rest_names = []
#     for dic in cursor:
#         rest_names.append(dic['rest_name'])
#     # Sampling 20 of them
#     sample = random.sample(set(rest_names), 20)
#     cursor = coll.find({'rest_name': {'$in': sample}})
#     df = pd.DataFrame(list(cursor))

#     print df.columns
#     print df.shape
#     print 'Chosen restaurants'
#     print df['rest_name'].unique()
#     df.to_pickle('../front_end/data/df_clean2.pkl')

def main3():
    df = pd.read_pickle('../front_end/data/df_clean2a.pkl')
    texts = df['review']

    n_topics = 100
    ngram_range = (2, 2)
    max_words = 5000
    max_iter = 400

    top_filename = '../../data/te_2_20_%d_%dgram_max_%d_100_extraStopW.txt' % \
                    (n_topics, ngram_range[1], max_words)

    model_filename = '../front_end/data/te_2a_extraSW.pkl'

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

def build_results(rest_name, base, base_fig):
    # paths have to be written from the front_end/app/ viewpoint
    df = pd.read_pickle(base + 'df_clean2a.pkl')
    texts = df['review']

    model_filename = base + 'te_2a_extraSW.pkl'
    te = dill.load(open(model_filename, 'rb'))
    categories = Categories()
    special = {'food', 'service', 'ambience'}

    sent_analyzer = BlobSentimentAnalysis()

    texts = df[df['rest_name'] == rest_name]['review'].values
    rid = df[df['rest_name'] == rest_name]['rid'].unique()[0]

    print 'Running top_cat...'
    tic = timeit.default_timer()
    top_cat = te.top_categories(texts, cat=categories)
    for c in special:
        if c not in top_cat:
            top_cat = np.append(top_cat, c)
    print top_cat
    tac = timeit.default_timer()
    print 'Finished top_cat in %.3f seconds' % (tac - tic)

    sentences = {}
    sentiments = {}
    print 'Looping over categories...'
    tic = timeit.default_timer()
    for c in top_cat:
        print c
        cloud_name = '%s_%s' % (rid, c)
        cloud_name = base_fig + cloud_name
        print cloud_name
        te.extract_onecat_topwords(texts, c, cloud_name, base_fig)
        sentences[c] = te.extract_onecat_sentences(texts, c, token=False)
        sentiments[c] = sent_analyzer.sentiment_sentences(sentences[c])
    tac = timeit.default_timer()
    print 'End looping in %.3f seconds' % (tac - tic)

    filename = '%s_snippets.pkl' % rid
    filename = base + filename
    tic = timeit.default_timer()
    with open(filename, 'w') as f:
        dill.dump(sentences, f)
    f.close()
    tac = timeit.default_timer()
    print 'Finished pickling sentences in %.3f seconds' % (tac - tic)

    filename = '%s_sentiments.pkl' % rid
    filename = base + filename
    tic = timeit.default_timer()
    with open(filename, 'w') as f:
        dill.dump(sentiments, f)
    f.close()
    tac = timeit.default_timer()
    print 'Finished pickling sentiments in %.3f seconds' % (tac - tic) 

    top_cat = [item for item in top_cat if item not in {'food', 'service', 
               'ambience'}]    

    return sentences, sentiments

def main4():
    rest_name = 'Il Borgo'
    base = '../front_end/data/'
    sentences, sentiments = build_results(rest_name)

    for cat, sent in sentences.iteritems():
        print cat
        print sent

    for cat, sent in sentiments.iteritems():
        print cat
        print sent 

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
    main4()
