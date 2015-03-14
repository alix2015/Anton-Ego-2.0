import numpy as np 
import pandas as pd 
from itertools import izip
import matplotlib.pyplot as plt 
from wordcloud import WordCloud
from textblob import TextBlob
# from textblob.sentiments import NaiveBayesAnalyzer
import dill
import timeit


class BlobSentimentAnalysis(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def sentiment(self, sentence):
        '''
        INPUT: BlobSentimentAnalysis object, string
        OUTPUT: float 2-tuple

        This method computes the polarity and subjectivity of sentence.
        Polarity ranges from -1. to +1., -1. being the most negative
        sentiment and +1. the most positive (happy).
        Subjectivity ranges from 0. to +1., 0. being the most objective,
        +1. the most subjective.
        Based on PatternAnalyzer http://www.clips.ua.ac.be/pattern
        '''
        #NaiveBayesAnalyzer way too slow: using PatternAnalyzer instead
        blob = TextBlob(sentence) 
        sentiment = blob.sentiment
        return sentiment.polarity, sentiment.subjectivity

    def sentiment_sentences(self, sentences, n=5):
        '''
        INPUT: BlobSentimentAnalysis object, list of strings, [integer]
        OUTPUT: list of 3-tuples of 2-tuples (score, sentence)

        This method performs sentiment analysis on each of the sentence
        of the given list of sentences.
        '''
        sentiment = []
        cnt = 0
        for sentence in sentences:
            sentiment.append(self.sentiment(sentence))
            if self.verbose:
                if not (cnt % 10):
                    print '%d sentences analysed' % (cnt + 1)
            cnt += 1
        
        sentiment = np.array(sentiment)

        #
        idx_pos = np.argsort(sentiment[0, :])[-1:-(n+1):-1]
        idx_neg = np.argsort(sentiment[0, :])[:n]
        idx_subj = np.argsort(sentiment[1, :])[-1:-(n+1):-1]
        pos = [sentences[i] for i in idx_pos]
        neg = [sentences[i] for i in idx_neg]
        subj = [sentences[i] for i in idx_subj]

        pos = [tup for tup in izip(sentiment[0, idx_pos], pos)]
        neg = [tup for tup in izip(sentiment[0, idx_neg], neg)]
        subj = [tup for tup in izip(sentiment[1, idx_subj], subj)]

        return [tup for tup in izip(pos, neg, subj)]
