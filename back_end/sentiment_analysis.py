import numpy as np 
import pandas as pd 
import re
import matplotlib.pyplot as plt 
# from collections import defaultdict
from nltk import RegexpParser
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
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

        if p_pos.shape[0] == 0:
            p_pos = np.zeros(1)
        if p_neg.shape[0] == 0:
            p_neg = np.zeros(1)
        
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
               

class BlobSentimentAnalysis(object):
    def __init__(self, analyzer=None, verbose=False):
        self.analyzer = analyzer
        self.verbose = verbose
        
    def sentiment(self, sentence):
        if self.analyzer:
            blob = TextBlob(sentence, analyzer=self.analyzer)
        else:
            blob = TextBlob(sentence)
        sentiment = blob.sentiment
        # return sentiment.p_pos, sentiment.p_neg Only for NaiveBayes
        return sentiment[0], sentiment[1]

    def sentiment_sentences(self, sentences):
        sentiment = []
        cnt = 0
        for sentence in sentences:
            sentiment.append(self.sentiment(sentence))
            if self.verbose:
                if not (cnt % 10):
                    print '%d sentences analysed' % (cnt + 1)
            cnt += 1
        sentiment = np.array(sentiment)

        return (sentiment.mean(axis=0), sentiment.max(axis=0),
                sentiment.min(axis=0))


class NPSentimentAnalysis(object):
    def __init__(self, sentWords):
        self.sentWords = sentWords
        self.grammar_np = r"""
                        NBAR:
                            {<NN.*|JJ.*>*<NN.*>} 
                            # Nouns and Adjectives, terminated with Nouns
                            
                        NP:
                            {<NBAR><IN|CC><NBAR>} 
                            # Above, connected with in/of/etc...
                            {<NBAR>}
                        """
        self.grammar_vp = r"""
                        NBAR:
                            {<NN.*|JJ>*<NN.*>}
                            
                        NP:
                            {<DT>*<NBAR><IN|CC><NBAR>}
                            {<DT>*<NBAR>}

                        VP:
                            {<NP>*<RB>*<VB.*><RB>*<VB.*>*<NP|JJ>*}
                        """

    # def get_word(self, pos_tokens, filter):
    #     for token in pos_tokens:
    #         if token[1] in filter:
    #             yield token[0]

    def dependencies_parser(self, sentence, kind='NP'):
        tokens = word_tokenize(sentence)
        if kind == 'NP':
            tree = RegexpTokenizer(self.grammar_np).parse(pos_tag(tokens))

        if kind == 'VP':
            tree = RegexpTokenizer(self.grammar_vp).parse(pos_tag(tokens))

    def majoritary_sentiment(self, scores):
        scores = np.array(scores)
        return scores[np.argsort(abs(scores))[-1]]

    def np_leaves(self, tree):
        for subtree in tree.subtrees(filter=lambda t: t.label()=='NP'):
            yield subtree.leaves()

    def np_sentiment(self, tree):
        sentiment = []
        for leaves in self.np_leaves(tree):
            s_n = []
            s_j = []
            s_r = []
            for token in leaves:
                if re.match(r'NN.*', token[1]):
                    s_n.append(self.sentWords.get_score(token[0]))
                if re.match(r'JJ.*', token[1]):
                    s_j.append(self.sentWords.get_score(token[0]))
                if re.match(r'RB', token[1]):
                    s_r.append(self.sentWords.get_score(token[0]))
            s_n = self.majoritary_sentiment(s_n)
            if s_j:
                s_j = self.majoritary_sentiment(s_j)
                if s_n < 0:
                    s_n = s_n - (1 + s_n) * abs(s_j)
                elif s_j > 0::
                    s_n = s_n + (1 - s_n) * s_j
                else:
                    s_n = s_a

            if s_r:
                s_r = self.majoritary_sentiment(s_r)
                s_n = - np.sign(s_n) * (abs(s_n) * (1 - abs(s_n)))
            sentiment.append(s_n)
        
        return self.majoritary_sentiment(sentiment)

    def vp_leaves(self, tree):
        for subtree in tree.subtrees(filter=lambda t: t.label()=='VP'):
            yield subtree.leaves()

    def vp_sentiment(self, tree):
        sentiment = []
        for leaves in self.vp_leaves(tree):
            s_v = []
            s_n = []
            s_r = []
            for token in leaves:
                if re.match(r'VV.*', token[1]):
                    s_r.appent(self.sentWords.get_score(token[0]))
                if re.match(r'NN.*', token[1]):
                    s_n.append(self.sentWords.get_score[0])
                if re.match(r'RB', token[1]):
                    s_r.append(self.sentWords.get_score[0])
            s_v = self.majoritary_sentiment(s_v)
            if s_n:
                s_n = self.majoritary_sentiment(s_n)
                if s_v > 0:
                    s_v = np.sign(s_n) * (abs(s_n) + (1 - abs(s_n)) * s_v)
                elif s_n < 0:
                    s_v = np.sign(s_n) * (abs(s_n) + (1 - abs(s_n)) * s_v)
            if s_r:
                s_r = self.majoritary_sentiment(s_r)
                if s_v < 0:
                    s_v = - (abs(s_v) + (1 - abs(s_v)) * abs(s_r))
                elif s_r > 0:
                    s_v = abs(s_v) + (1 - abs(s_v)) * abs(s_r)
                else:
                    s_v = s_r
            sentiment.append(s_v)

        return self.majoritary_sentiment(sentiment)