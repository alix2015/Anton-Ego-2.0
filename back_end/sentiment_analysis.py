import numpy as np 
import pandas as pd 
import re
import matplotlib.pyplot as plt 
# from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from wordcloud import WordCloud
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import dill
import timeit



class SentimentWords(object):
    def __init__(self, source):
        # self.destination = psycopg2.connect(dbname=destination,
        #                                     user='postgres', host='/tmp')
        self.table = None
        self._parse_source(source)
        self._parse_terms()
        self.dic = {}
        self._build_dic()

    def _parse_source(self, source):
        items = []
        with open(source, 'r') as f:
            for line in f:
                items.append(line.split('\t'))
        f.close()
        # Values of interest lines 27 to 117,687 (ignoring last line)
        self.table = pd.DataFrame(items[27:-2])
        # Column names in line 26
        self.table.columns = items[26]
        # Converting scores into float
        self.table['PosScore'] = self.table['PosScore'].astype(float)
        self.table['NegScore'] = self.table['NegScore'].astype(float)

    def _parse_terms(self):
        pattern = re.compile(r'\w+')
        self.table['terms'] = self.table['SynsetTerms']\
                                .map(lambda s: s.split())\
                                .map(lambda l: [pattern.match(w).group(0) if 
                                pattern.match(w) else '' for w in l])

    def _build_dic(self):
        for idx in self.table.index:
            for w in self.table.ix[idx]['terms']:
                self.dic[w] = (self.table.ix[idx]['PosScore'],
                               self.table.ix[idx]['NegScore'])

    def get_score(self, word):
        return self.dic.get(word, (0., 0.))


class BlobSentimentAnalysis(object):
    def __init__(self, sent):
        self.blob = TextBlob(sent, analyzer=NaiveBayesAnalyzer())

    def sentiment(self):
        sent = self.blob.sentiment
        return sent.p_pos, sent.p_neg 

class AvgSentimentAnalysis(object):
    def __init__(self, sentWords, sent):
        self.words = sent.split()
        self.sentWords = sentWords

    # def pos_tag(self, sent):
    #     return pos_tag(sent.split())

    def sentiment(self):
        probas = np.array([[self.sentWords.get_score(w)[0],
                          self.sentWords.get_score(w)[1]] \
                          for w in self.words if w not in stopwords.words()])
        return probas.mean(axis=0), probas.max(axis=0), probas.min(axis=0)



class SentimentAnalysis(object):
    """docstring for SentimentAnalysis"""
    def __init__(self, rest_names, sentWord):
        self.rest_names = rest_names
        self.priors = sentWord.get_score


    # def setting_priors():

