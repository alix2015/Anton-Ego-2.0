from itertools import izip
from textblob import TextBlob
# from textblob.sentiments import NaiveBayesAnalyzer
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
        return round(sentiment.polarity, 2), round(sentiment.subjectivity, 2)

    def sentiment_sentences(self, sentences, n=5):
        '''
        INPUT: BlobSentimentAnalysis object, list of strings, [integer]
        OUTPUT: list of dictionaries (polarity, subjectivity, sentence)

        This method performs sentiment analysis on each of the sentence
        of the given list of sentences.
        '''
        sentiments = []
        cnt = 0
        for sentence in sentences:
            sentiments.append(self.sentiment(sentence))
            if self.verbose:
                if not (cnt % 10):
                    print '%d sentences analysed' % (cnt + 1)
            cnt += 1
        # Sort in decreasing order of sentiment
        answ = [tup for tup in izip(sentiments, sentences)]
        answ.sort(key=lambda t: -t[0][0])
        return answ
        