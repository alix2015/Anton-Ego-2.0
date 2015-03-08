import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
# from sklearn.cluster import KMeans
from wordcloud import WordCloud
from sentiment_analysis import BlobSentimentAnalysis, AvgSentimentAnalysis
# from sklearn.metrics import precision_score, recall_score, accuracy_score
# from sklearn.pipeline import Pipeline
# import cPickle as pickle
import dill




class TopicExtraction(object):
    def __init__(self,
                 rest_names=[],
                 n_topics=6,
                 sentence=False,
                 ngram_range=(1, 1),
                 max_words=None,
                 max_iter=200):
        '''
        INPUT: TopicExtraction object, <list of strings, integer, boolean,
        tuple of integers, integer, integer>
        OUTPUT: None

        rest_names are added to the stopwords
        sentence is a boolean indicating whether sentences are the documents
        '''
        self.rest_names = rest_names
        self.n_topics = n_topics
        self.sentence = sentence
        self.max_words = max_words
        self.ngram_range = ngram_range
        self.stopwords = stopwords.words()
        self.max_iter = max_iter
        self.vectorizer = TfidfVectorizer(stop_words='english',
                                          max_features=self.max_words,
                                          tokenizer=self.my_tokenize,
                                          ngram_range=self.ngram_range)
        self.factorizor = NMF(n_components=self.n_topics,
                              max_iter=self.max_iter)
        self.trained = False
        self.H_ = None
        self.category = None
        self.fonts_ = '/Library/Fonts/Georgia.ttf'
        self.wordcloud = WordCloud(font_path=self.fonts_)
        

    def my_tokenize(self, text):
        '''
        INPUT: TopicExtraction object, string
        OUTPUT: list of strings

        tokenizing
        '''
        text = text.lower().encode('ascii', errors='ignore')
        rest_names = ' '.join(self.rest_names).lower()\
                    .encode('ascii', errors='ignore')
        # list_tokenized = RegexpTokenizer(r'\w+').tokenize(text)
        list_tokenized = RegexpTokenizer(r'\W\s+|\s+\W|\W+\b|\b\W+',
                                         gaps=True).tokenize(text)
        rest_names = RegexpTokenizer(r'\W\s+|\s+\W|\W+\b|\b\W+',
                                     gaps=True).tokenize(rest_names)
        for name in rest_names:
            self.stopwords.append(name)
        list_tokenized = [word for word in list_tokenized\
                            if word not in self.stopwords]
        
        return list_tokenized

    def fit_transform(self, texts):
        '''
        INPUT: TopicExtraction object, list of strings
        OUTPUT: array
        vectorizing and factorizing into n_topics latent topics
        '''
        if self.sentence:
            texts = [sent for item in
                    [sent_tokenize(text) for text in texts] for
                    sent in item] 
        V = self.vectorizer.fit_transform(texts)
        W = self.factorizor.fit_transform(V)
        self.H_ = self.factorizor.components_
        self.trained = True
        
        return W

    def extract_top_words(self,
                          texts,
                          top_n=10,
                          top_filename=None,
                          wordcloud=False):
        '''
        INPUT: list of strings, integer <string, boolean>
        OUTPUT: list of strings

        This method trains the model (vectorizer and factorizer);
        it extract the top_n top words per latent topic.
        Optionally, export to a file the list of words per latent topic.
        Optionally, export the top words as word clouds.
        '''
        if not self.trained:
            self.fit_transform(texts)
        top_words = {}
        for topic in xrange(self.n_topics):
            top_words_idx = np.argsort(self.H_[topic, :])[-1:-(top_n + 1):-1]
            top_words[topic] = [self.vectorizer.get_feature_names()[i] for
                                i in top_words_idx]
            if wordcloud:
                self.cloud_fig(top_words[topic],
                               '../data/nouncloud_%d.png' % topic)
        
        if top_filename:
            with open(top_filename, 'w') as f:
                f.write('n_gram: %d, %d' % (ngram_range[0], ngram_range[1]))
                f.write('\n')
                f.write('n_topics: %d' % n_topics)
                f.write('\n')
                f.write('-------------------------')
                f.write('\n')
                for topic in top_words:
                    f.write('Topic %d' % topic)
                    f.write('\n')
                    for word in top_words[topic]:
                        f.write(word)
                        f.write('\n')
                    f.write('-------------------------')
                    f.write('\n')
            f.close()

        return top_words

    def cloud_fig(self, top_words, filename):
        '''
        INPUT: TopicExtraction object, list of strings, string
        OUTPUT: None

        Given a list of words, builds a word cloud and save the figure
        in filename.
        '''
        wordcloud = self.wordcloud.generate(' '.join(top_words))
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.savefig(filename)
        plt.close()
        
    def _define_categories(self, dic):
        '''
        INPUT: TopicExtraction object, dictionary
        OUTPUT: None

        Updates the category attribute of the TopicExtraction object,
        with a dictionary the keys of which are the latent topic name
        and the value is a set of the indices of the corresponding latent
        topic components in the factorization.
        So far, hand-labelling of the latent topics
        Possible improvement: using an ontology
        '''
        self.category = dic
        # print 'Categories created'

    def extract_onecat_topwords(self,
                                texts,
                                category,
                                filename,
                                top_n=15,
                                dic=None):
        '''
        INPUT: TopicExtraction object, list of strings, string, 
               string, integer, <dictionary>
        OUTPUT: list of strings

        This method transforms a test set using the trained model
        to extract the top words in the latent topics corresponding
        to one category. In case the category attribute of the TopicExtraction
        object has not yet been initialized, a dictionary should be provided.
        It exports these words as a word cloud in filename and returns them.
        '''

        if not self.category:
            if dic:
                self._define_categories(dic)
            else:
                print 'Please provide a dictionary to initialize the categories'
                return
        if self.sentence:
            texts = [sent for item in
                    [sent_tokenize(text) for text in texts] for
                    sent in item]
        V = self.vectorizer.transform(texts)
        W = self.factorizor.transform(V)

        top_words = []
        for topic in self.category[category]:
            top_doc_idx = np.argsort(W[:, topic])[-1:-(top_n + 1):-1]
            temp = [self.my_tokenize(texts[idx]) for idx in top_doc_idx]
            temp = [item for sublist in temp for item in sublist]
            
            for item in temp:
                top_words.append(item)
        
        self.cloud_fig(top_words, '../../data/%s.png' % filename)
        return top_words


    def extract_onecat_sentences(self, texts, category, dic=None):
        '''
        INPUT: TopicExtraction object, list of strings, string, <dictionary>
        OUTPUT: list of list of strings

        This method extracts from a test set of documents the sentences
        relevant to the given category. If the category attribute of
        the TopicExtraction object has not been initialized a dictionary
        should be provided.
        '''
        if not self.category:
            if dic:
                self._define_categories(dic)
            else:
                print 'Please provide a dictionary to initialize the categories'
                return
        texts = [sent for item in [sent_tokenize(text) for text in texts]\
                 for sent in item]
        V = self.vectorizer.transform(texts)
        W = self.factorizor.transform(V)

        for topic in self.category[category]:
            idx = np.argsort(W[:, topic])[-1:-16:-1]
            docs = [self.my_tokenize(texts[i]) for i in idx]

        return docs

