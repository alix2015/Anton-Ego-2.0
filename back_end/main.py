import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from processing_data import TopicExtraction
from sentiment_analysis import SentimentWords, AvgSentimentAnalysis
import dill
import timeit

'''
This file is used for the ongoing running and testing.
It will be cleaned at the end and be used as an example file
on how to use this package.
'''

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

    te = TopicExtraction(rest_names=rest_names,
                         n_topics=n_topics,
                         sentence=True,
                         ngram_range=ngram_range,
                         max_words=max_words,
                         max_iter=max_iter)


    if top_filename:
        top_words = te.extract_top_words(df['reviews'],
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

def categorize():
    dic = {}
    dic['food'] = {4, 6, 7, 21, 24, 32, 34, 35, 2, 36, 25, 37,
                            38, 39, 40, 41, 42, 44, 47, 48, 53, 54, 55,
                            56, 65, 68, 69, 70, 73, 75, 79, 85, 91, 93,
                            95, 97}
    dic['service'] = {5, 9, 15, 18, 19, 45, 58, 60, 62, 66, 71,
                                72, 81, 90, 94, 98}
    dic['ambience'] = {10, 16, 20, 26, 31, 61, 64}
    
    dic['wine'] = {2, 36}
    dic['cocktail'] = {25, 36}
    dic['steak'] = {21}
    dic['Chinese'] = {24}
    dic['French'] = {47, 87}
    dic['cheese'] = {34, 97}
    dic['dessert'] = {35, 40, 95}
    dic['vegetables'] = {37}
    dic['meat'] = {42, 44, 47, 49, 51, 53, 69, 70, 85, 91}
    dic['pork'] = {51}
    dic['steak'] = {49, 53, 69, 70, 85}
    dic['egg'] = {44}
    dic['potato'] = {44, 48}
    dic['entree'] = {38, 39}
    dic['layout'] = {16, 26}
    dic['noise'] = {17, 64, 82}
    dic['music'] = {64, 82}
    dic['location'] = {26, 50, 52, 76, 77}
    dic['vegetarian'] = {56, 87}
    dic['salad'] = {87}
    dic['brunch'] = {65, 90}
    dic['Mediterranean'] = {73}
    dic['Indian'] = {79}
    
    dic['excellent'] = {3, 5, 18, 20, 25, 27, 29, 33, 34, 96, 99}
    dic['positive sentiment'] = {9, 10, 11, 12, 15, 19, 
                                 22, 28, 45, 46, 54, 59, 60, 62,
                                 63, 66, 68, 80, 81, 86, 90, 94}
    dic['negative sentiment'] = {46, 58, 71, 94}
    dic['experience'] = {8, 78, 92}
    dic['positive recommendation'] = {13, 23, 30, 74, 83}
    dic['special occasion'] = {14, 31, 43, 59, 74, 84, 89}
    dic['reservation'] = {60}
    dic['price'] = {67}
    dic['cook'] = {68, 75}

    return dic


def main1():
    data_SF = '../../data/reviews_SF.pkl'
    data_1 = '../../data/reviews_1.pkl'
    data_2 = '../../data/reviews_2.pkl'

    df =  build_data([data_SF, data_1, data_2], min_rev_len=100)

    n_topics = 100
    ngram_range = (2, 2)
    max_words = 5000
    max_iter = 400

    top_filename = '../../data/topics_5_%d_%dgram_max_%d_100_s.txt' % \
                    (n_topics, ngram_range[1], max_words)

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
    
    dic = categorize()

    df_SF = pd.read_pickle(data_SF)
    texts = df[df['rest_name'] == '/absinthe-brasserie-and-bar']['reviews']
    tic = timeit.default_timer()
    te.extract_onecat_topwords(texts, 'wine', 'test5_wine', dic=dic)
    toc = timeit.default_timer()
    print 'One category highlights in %3.f seconds' % (toc - tic)
    te.extract_onecat_topwords(texts, 'experience', 'test5_experience')

    # te = build_model(df, n_topics, ngram_range, max_words, max_iter, 
    #                  top_filename, model_filename)


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


# def main5():



   
if __name__ == '__main__':
    main4()