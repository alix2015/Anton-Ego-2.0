import numpy as np 
import pandas as pd 
import re
import matplotlib.pyplot as plt 
# from collections import defaultdict
from nltk import RegexpParser
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from wordcloud import WordCloud
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import dill
import timeit



class SentimentWords(object):
    '''
    This class provides a lexicon of labelled sentiment words from
    SentiWordNet (http://sentiwordnet.isti.cnr.it).
    It has one accessible method, get_score, that yields the sentiment
    of a word as a tuple (positive, negative), where each sentiment is
    a float between 0. and 1.
    '''
    def __init__(self, source):
        self.table = None
        self.stemmer = PorterStemmer()
        self._parse_source(source)
        self._parse_terms()
        self.dic = {}
        self._build_dic()
        

    def _parse_source(self, source):
        '''
        INPUT: SentimentWords object, string
        OUTPUT: None

        Internal method used to extract the sentiment words and
        their sentiment from the text file source. Parsed data is
        stored in the dataframe table. 
        '''
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
        '''
        INPUT: SentimentWords object
        OUTPUT: None

        Internal method to parse the sentiment terms.
        '''
        pattern = re.compile(r'\w+')
        self.table['terms'] = self.table['SynsetTerms']\
                                .map(lambda s: s.split())\
                                .map(lambda l: [pattern.match(w).group(0) if 
                                pattern.match(w) else '' for w in l])

    def _build_dic(self):
        '''
        INPUT: SentimentWords object
        OUTPUT: None

        Internal method converting the table into a dictionary
        the keys of which are sentiment words and the values a
        tuple of float indicating the sentiment (positive, negative).
        '''
        for idx in self.table.index:
            for w in self.table.ix[idx]['terms']:
                w = self.stemmer.stem(w)
                self.dic[w] = (self.table.ix[idx]['PosScore'],
                               self.table.ix[idx]['NegScore'])

    def get_score(self, word):
        '''
        INPUT: SentimentWords object, string
        OUTPUT: tuple of float

        This method returns the sentiment corresponding to word
        as a tuple of float (positive, negative).
        '''
        w = self.stemmer.stem(word)
        return self.dic.get(w, (0., 0.))


class AvgSentimentAnalysis(object):
    def __init__(self, sentWords):
        self.priors = sentWords.get_score

    
    def sentiment_one_sent(self, tokens):
        '''
        INPUT: AvgSentimentAnalysis object, list of strings
        OUTPUT: 3-tuple of 2-list

        This method returns the average, max, and min sentiment
        of a sentence given as a list of tokens.
        '''
        probas = []
        for token in tokens:
            p_pos, p_neg = self.priors(token)
            
            probas.append([p_pos, p_neg])
        
        probas = np.array(probas)
        p_pos = probas[probas[:, 0] > 0][:, 0]
        p_neg = probas[probas[:, 1] > 0][:, 1]
        
        return ([p_pos.mean(axis=0), p_neg.mean(axis=0)],
                [p_pos.max(axis=0), p_neg.max(axis=0)], 
                [p_pos.min(axis=0), p_neg.min(axis=0)])
        

    def sentiment_sentences(self, sentences):
        '''
        INPUT: AvgSentimentAnalysis object, list of list of strings
        OUTPUT: 3-tuple of (2,) numpy array

        This method returns the average, max, and min sentiment
        of a list of sentences given each as a list of tokens.
        '''
        avg_sentiment = []
        max_sentiment = []
        min_sentiment = []
        for token_list in sentences:
            tup = self.sentiment_one_sent(token_list)
            avg_sentiment.append(tup[0])
            max_sentiment.append(tup[1])
            min_sentiment.append(tup[2])
        avg_sentiment = np.array(avg_sentiment)
        max_sentiment = np.array(max_sentiment)
        min_sentiment = np.array(min_sentiment)
        
        return (avg_sentiment.mean(axis=0), 
                max_sentiment.mean(axis=0),
                min_sentiment.mean(axis=0))
               

