import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from categories import Categories
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
# from sklearn.cluster import KMeans
from wordcloud import WordCloud
from sentiment_analysis import AvgSentimentAnalysis
# from sklearn.metrics import precision_score, recall_score, accuracy_score
# from sklearn.pipeline import Pipeline
# import cPickle as pickle
import dill
import timeit




class TopicExtraction(object):
    def __init__(self,
                 rest_names=[],
                 n_topics=6,
                 sentence=False,
                 ngram_range=(1, 1),
                 max_words=None,
                 max_iter=200,
                 categories=None,
                 verbose=False):
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
        self.verbose = verbose
        self.vectorizer = TfidfVectorizer(stop_words='english',
                                          max_features=self.max_words,
                                          tokenizer=self.my_tokenize,
                                          ngram_range=self.ngram_range)
        self.factorizor = NMF(n_components=self.n_topics,
                              max_iter=self.max_iter)
        self.trained = False
        self.H_ = None
        self.categories = categories
        self.fonts_ = '/Library/Fonts/Georgia.ttf'
        self.wordcloud = WordCloud(font_path=self.fonts_)
        

    def my_tokenize(self, text):
        '''
        INPUT: TopicExtraction object, string
        OUTPUT: list of strings

        tokenizing
        '''
        text = text.lower().encode('ascii', errors='ignore')
        # list_tokenized = RegexpTokenizer(r'\w+').tokenize(text)
        list_tokenized = RegexpTokenizer(r'\W\s+|\s+\W|\W+\b|\b\W+',
                                         gaps=True).tokenize(text)
        if self.rest_names:
            if self.verbose:
                tic = timeit.default_timer()
            rest_names = ' '.join(self.rest_names).lower()\
                        .encode('ascii', errors='ignore')
            rest_names = RegexpTokenizer(r'\W\s+|\s+\W|\W+\b|\b\W+',
                                         gaps=True).tokenize(rest_names)
            for name in rest_names:
                self.stopwords.append(name)
            # Adding positive adjectives/verbs that dominate reviews
            self.stopwords.append('good')
            self.stopwords.append('excellent')
            self.stopwords.append('great')
            self.stopwords.append('friendly')
            self.stopwords.append('nice')
            self.stopwords.append('decent')
            self.stopwords.append('say')
            self.stopwords.append('recommend')
            self.stopwords.append('like')
            self.stopwords.append('love')
            self.stopwords.append('recommended')
            self.stopwords.append('liked')
            self.stopwords.append('loved')
            self.stopwords.append('dining')
            # Adding genering words
            self.stopwords.append('restaurant')
            self.stopwords.append('restaurants')
            self.stopwords.append('food')
            self.stopwords.append('ambience')
            self.stopwords.append('service')
            self.stopwords.append('dining')
            self.rest_names = None # TO AVOID REPROCESSING
            self.stopwords = set(self.stopwords)
            if self.verbose:
                tac = timeit.default_timer()
                print 'Adding restaurants in %d seconds' % (tac - tic)
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
        if self.verbose:
            tic = timeit.default_timer()
            print 'Starting Tfidf vectorizer...'
        V = self.vectorizer.fit_transform(texts)
        if self.verbose:
            toc = timeit.default_timer()
            print 'Finished vectorizing in %.3f' % (toc - tic)
            tic = timeit.default_timer()
            print 'Starting factorizing...'
        W = self.factorizor.fit_transform(V)
        if self.verbose:
            toc = timeit.default_timer()
            print 'Finished vectorizing in %3.f' % (toc - tic)
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
                f.write('n_gram: %d, %d' % (self.ngram_range[0], 
                        self.ngram_range[1]))
                f.write('\n')
                f.write('n_topics: %d' % self.n_topics)
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
        
    def _define_categories(self, categories):
        '''
        INPUT: TopicExtraction object, dictionary
        OUTPUT: None

        Updates the categories attribute of the TopicExtraction object,
        with a dictionary the keys of which are the latent topic name
        and the value is a set of the indices of the corresponding latent
        topic components in the factorization.
        So far, hand-labelling of the latent topics
        Possible improvement: using an ontology
        '''
        self.categories = categories
        # print 'Categories created'

    def extract_onecat_topwords(self,
                                texts,
                                category,
                                cloud=False,
                                filename,
                                base='',
                                top_n=5,
                                cat=None):
        '''
        INPUT: TopicExtraction object, list of strings, string, 
               string, integer, <categories>
        OUTPUT: list of strings

        This method transforms a test set using the trained model
        to extract the top words in the latent topics corresponding
        to one category. In case the category attribute of the TopicExtraction
        object has not yet been initialized, a dictionary should be provided.
        It exports these words as a word cloud in filename and returns them.
        '''

        if not self.categories:
            if cat:
                self._define_categories(cat)
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
        for topic in self.categories.get(category):
            top_doc_idx = np.argsort(W[:, topic])[-1:-(top_n + 1):-1]
            temp = [self.my_tokenize(texts[idx]) for idx in top_doc_idx]
            temp = [item for sublist in temp for item in sublist
                    if item not in self.stopwords]
            
            for item in temp:
                top_words.append(item)
        
        self.cloud_fig(top_words, '%s.png' % filename)
        return top_words


    def extract_onecat_sentences(self, texts, category, cat=None, token=True):
        '''
        INPUT: TopicExtraction object, list of strings, string, 
        <categories, boolean>
        OUTPUT: list of list of strings

        This method extracts from a test set of documents the sentences
        relevant to the given category. If the category attribute of
        the TopicExtraction object has not been initialized a dictionary
        should be provided. If token, sentences are tokenized, otherwise
        they are returned as are.
        '''
        if not self.categories:
            if cat:
                self._define_categories(cat)
            else:
                print 'Please provide a dictionary to initialize the categories'
                return

        texts = [sent for item in [sent_tokenize(text) for text in texts]\
                 for sent in item]
        V = self.vectorizer.transform(texts)
        W = self.factorizor.transform(V)

        for topic in self.categories.get(category):
            idx = np.argsort(W[:, topic])[-1:-6:-1]
            if token:
                docs = [self.my_tokenize(texts[i]) for i in idx]
            else:
                docs = [texts[i] for i in idx]

        return docs

    def top_categories(self, texts, top_n=5, cat=None):

        if not self.categories:
            if cat:
                self._define_categories(cat)
            else:
                print 'Please provide a dictionary to initialize the categories'
                return

        # What are the most relevant categories?
        # The ones that yield the highest largest projections on
        # the category subspace
        
        texts = [sent for item in [sent_tokenize(text) for text in texts]\
                 for sent in item]
        V = self.vectorizer.transform(texts)
        W = self.factorizor.transform(V)

        max_coef = np.zeros(len(self.categories))
        idx = 0
        names = []
        for category in self.categories:
            names.append(category)
            for topic in self.categories.get(category):
                m = max(W[:, topic])
                if m > max_coef[idx]:
                    max_coef[idx] = m
            idx += 1

        names = np.array(names)
        return names[np.argsort(max_coef)[-1:-(top_n+1):-1]]




