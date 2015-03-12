import numpy as np 
import pandas as pd 
from pymongo import MongoClient
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
# from textblob.sentiments import NaiveBayesAnalyzer
from processing_data import TopicExtraction
from categories import Categories
from sentiment_analysis import SentimentWords, AvgSentimentAnalysis
from sentiment_analysis import BlobSentimentAnalysis
import dill
import timeit
from collections import defaultdict
import random

'''
This file is used for the ongoing running and testing.
It will be cleaned at the end and be used as an example file
on how to use this package.
'''


# Utilities for a pipeline based on MongoDB

def build_model_mongo(n_topics,
                      ngram_range,
                      max_words,
                      max_iter,
                      model_filename,
                      verbose=False,
                      fraction=None,
                      top_filename=None):
    
    client = MongoClient()
    coll = client.opentable.clean3

    rest_names = coll.distinct('rest_name')

    te = TopicExtraction(rest_names=rest_names,
                         n_topics=n_topics,
                         sentence=True,
                         ngram_range=ngram_range,
                         max_words=max_words,
                         max_iter=max_iter,
                         verbose=verbose)

    # Restricting to review longer than 150 characters
    cursor = coll.find({'review_length': {'$gt': 150}}, {'review': 1, '_id': 0})
    reviews = []

    cnt = 0
    for dic in cursor:
        reviews.append(dic['review'])
        cnt += 1
        if not (cnt % 50):
            print '%d reviews read' % (cnt + 1)

    if fraction:
        n_rev = int(fraction * len(reviews))
        idx = np.random.random_integers(0, len(reviews) - 1, n_rev)
        reviews = [reviews[i] for i in idx]
        print 'Subset of %d reviews' % n_rev

    if top_filename:
        top_words = te.extract_top_words(reviews,
                                         top_n=15,
                                         top_filename=top_filename)
                                         # wordcloud=True)
    else:
        te.fit_transform(reviews)

    if model_filename:
        with open(model_filename, 'w') as f:
            dill.dump(te, f)
            print 'Finished pickling the model'
    return te


def main2():
    source = '../../data/sentiWords/SentiWordNet_3.0.0_20130122.txt'
    tic = timeit.default_timer()
    sentWord = SentimentWords(source)
    toc = timeit.default_timer()
    print 'sentWord instanciation in %3.f seconds' % (toc - tic)
    sentWord_filename = '../../data/sentWord_2.pkl'
    with open(sentWord_filename, 'w') as f:
        dill.dump(sentWord, f)
        tic = timeit.default_timer()
        print 'Finished pickling the model in %3.f seconds' % (tic - toc)

    print 'terrible'
    print sentWord.get_score('terrible')
    print 'terribl'
    print sentWord.get_score('terribl')


def main3():
    data_SF = '../../data/reviews_SF.pkl'
    data_1 = '../../data/reviews_1.pkl'
    data_2 = '../../data/reviews_2.pkl'

    df =  build_data([data_SF, data_1, data_2], min_rev_len=100)

    n_topics = 100
    ngram_range = (2, 2)
    max_words = 5000
    max_iter = 400

    model_filename = '../../data/topics_5.pkl'

    tic = timeit.default_timer()

    te = build_model(df,
                     n_topics,
                     ngram_range,
                     max_words,
                     max_iter,
                     model_filename)

    toc = timeit.default_timer()

    print 'Building model in %3.f seconds' % (toc - tic)

    texts = df[df['rest_name'] == '/absinthe-brasserie-and-bar']['reviews']
    categories = ['food', 'wine', 'special occasion', 'price']

    sentWords = dill.load(open('../../data/sentWord.pkl', 'rb'))

    dic = categorize()

    avg_sent = AvgSentimentAnalysis(sentWords)

    for cat in categories:
        print 'Category: %s' % cat
        tic = timeit.default_timer()
        # av, ma, mi = te.sentiment_one_cat(sentWords, texts, cat)
        sentences = te.extract_onecat_sentences(texts, cat, dic)
        toc = timeit.default_timer()
        print 'Extracted sentences in %3.f' % (toc - tic)

        tic = timeit.default_timer()
        av, ma, mi = avg_sent.sentiment_sentences(sentences)
        toc = timeit.default_timer()
        print 'Sentiment analysis in %3.f' % (toc - tic)

        # print 'Sentiment: %d' % (p_pos > p_neg)
        print 'p_pos: %3.f, %3.f, %3.f' % (av[0], ma[0], mi[0])
        print 'p_neg: %3.f, %3.f, %3.f' % (av[1], ma[1], mi[1])



def main4():
    model_filename = '../../data/topics_5.pkl'

    te = dill.load(open(model_filename, 'rb'))

    data_SF = '../../data/reviews_SF.pkl'
    data_1 = '../../data/reviews_1.pkl'
    data_2 = '../../data/reviews_2.pkl'

    df =  build_data([data_SF, data_1, data_2], min_rev_len=100)

    texts = df[df['rest_name'] == '/absinthe-brasserie-and-bar']['reviews']
    categories = ['food', 'wine', 'special occasion', 'price']

    sentWords = dill.load(open('../../data/sentWord_2.pkl', 'rb'))

    dic = categorize()

    avg_sent = AvgSentimentAnalysis(sentWords)

    for cat in categories:
        print 'Category: %s' % cat
        tic = timeit.default_timer()
        # av, ma, mi = te.sentiment_one_cat(sentWords, texts, cat)
        sentences = te.extract_onecat_sentences(texts, cat, dic)
        toc = timeit.default_timer()
        print 'Extracted sentences in %3.f' % (toc - tic)

        tic = timeit.default_timer()
        av, ma, mi = avg_sent.sentiment_sentences(sentences)
        toc = timeit.default_timer()
        print 'Sentiment analysis in %3.f' % (toc - tic)

        # print 'Sentiment: %d' % (p_pos > p_neg)
        print 'p_pos: %3.f, %3.f, %3.f' % (av[0], ma[0], mi[0])
        print 'p_neg: %3.f, %3.f, %3.f' % (av[1], ma[1], mi[1])


def main5():
    n_topics = 100
    ngram_range = (2, 2)
    max_words = 5000
    max_iter = 400
    fraction = .1

    top_filename = 'te_3_%d_%dgram_max_%d_100_s.txt' % \
                    (n_topics, ngram_range[1], max_words)

    model_filename = '../front_end/data/te_3.pkl'

    verbose = True

    tic = timeit.default_timer()

    te = build_model_mongo(n_topics,
                           ngram_range,
                           max_words,
                           max_iter,
                           model_filename,
                           verbose,
                           fraction,
                           top_filename)

    toc = timeit.default_timer()

    print 'Building model in %3.f seconds' % (toc - tic)

def sentiment_texts(reviews, text=True):
    
    blob_sent = BlobSentimentAnalysis(verbose=True)
    # NaiveBayesAnalyzer very slow
    # blob_sent = BlobSentimentAnalysis(analyzer=NaiveBayesAnalyzer,
    #                                   verbose=True)

    if text: 
        reviews = [item for review in reviews for item in
                     sent_tokenize(review)]
    
    tic = timeit.default_timer()
    sentiment = blob_sent.sentiment_sentences(reviews)
    toc = timeit.default_timer()
    # print 'Sentiment'
    # print sentiment
    print 'Duration: %.3f' % (toc - tic)

    return sentiment

def main6():
    '''
    Testing BlobSentimentAnalysis using a model trained on a small
    subset of data.
    '''

    client = MongoClient()
    coll = client.opentable.clean2

    model_path = '../../data/topics_5.pkl'
    te = dill.load(open(model_path, 'rb'))

    dic = categorize()
    te._define_categories(dic)

    results_path = '../../data/sentiments_main6.txt'
    
    # Test with a random restaurant: 'Mezcal' (127 reviews)
    # coll.find({'rest_name': 'Mezcal'}, {}).count()
    # Problem with this restaurant: no review saved?!?
    # Random sample of restaurants
    rest_reviews = defaultdict(list)
    cursor = coll.find({'review_length': {'$gt': 150}}, 
                       {'rest_name': 1, 'review': 1})
    for dic in cursor:
        name = dic['rest_name'].encode('ascii', errors='ignore')
        review = dic['review'].encode('ascii', errors='ignore')
        rest_reviews[name].append(review)
    rest_sample = random.sample(set(rest_reviews.keys()), 20)
    categories = ['food', 'service', 'ambience', 'special occasion',
                  'excellent']

    
    with open(results_path, 'w') as f:
        big_tic = timeit.default_timer()
        for resto in rest_sample:
            cursor = coll.find({'rest_name': resto}, {'review': 1, '_id': 0})
            reviews = []
            for dic in cursor:
                reviews.append(dic['review'])
            f.write('%d reviews for %s' % (len(reviews), resto))

            # Restricting to longer reviews
            lg_reviews = [review for review in reviews if len(review) > 100]
            f.write('%d long reviews for %s' % (len(lg_reviews), resto)) 

            # Restricting to categories
            for category in categories:
                tic = timeit.default_timer()
                sentences = te.extract_onecat_sentences(reviews, category, dic,
                                                        token=False)
                tac = timeit.default_timer()
                if sentences:
                    f.write('%d sentences relevant for category %s' % 
                            (len(sentences), category))
                    sentiments = sentiment_texts(sentences)
                else:
                    f.write('No corresponding sentence.')

                toc =timeit.default_timer()
                f.write('Summary for %s:' % category)
                f.write('All %d reviews in %.3f + %.3f seconds' %
                        (len(sentences), tac - tic, toc - tac))
                f.write('Positive')
                for i in xrange(len(sentiments[0])):
                    f.write(str(sentiments[0][i]))
                f.write('Negative')
                for i in xrange(len(sentiments[1])):
                    f.write(str(sentiments[1][i]))
                f.write('Subjective')
                for i in xrange(len(sentiments[2])):
                    f.write(str(sentiments[2][i]))

                tic = timeit.default_timer()
                sentences = te.extract_onecat_sentences(lg_reviews, category,
                                                        token=False)
                tac = timeit.default_timer()
                if sentences:
                    f.write('%d sentences relevant for category %s' %
                            (len(sentences), category))
                    sentiments = sentiment_texts(sentences)
                else:
                    f.write('No corresponding sentence.')
                toc = timeit.default_timer()
                f.write('Summary for %s:' % category)
                f.write('%d long reviews in %.3f + %.3f seconds' %
                        (len(sentences), tac - tic, toc - tac))
                f.write('Positive')
                for i in xrange(len(sentiments[0])):
                    f.write(str(sentiments[0][i]))
                f.write('Negative')
                for i in xrange(len(sentiments[1])):
                    f.write(str(sentiments[1][i]))
                f.write('Subjective')
                for i in xrange(len(sentiments[2])):
                    f.write(str(sentiments[2][i]))
        big_toc = timeit.default_timer()
        f.write('Total duration %.3f' % (big_toc - big_tic))
    f.close()

   
if __name__ == '__main__':
    main6()
