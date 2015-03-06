import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from sentiment_analysis import BlobSentimentAnalysis, AvgSentimentAnalysis
# from sklearn.metrics import precision_score, recall_score, accuracy_score
# from sklearn.pipeline import Pipeline
# import cPickle as pickle
import dill




class TopicExtraction(object):
    def __init__(self,
                 n_topics=6,
                 sentence=False,
                 ngram_range=(1, 1),
                 max_words=None,
                 max_iter=200):
        self.n_topics = n_topics
        self.sentence = sentence
        self.max_words = max_words
        self.ngram_range = ngram_range
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
        Input: string
        Output: list of strings
        '''
        text = text.lower().encode('ascii', errors='ignore')
        # list_tokenized = RegexpTokenizer(r'\w+').tokenize(text)
        list_tokenized = RegexpTokenizer(r'\W\s+|\s+\W|\W+\b|\b\W+',
                                         gaps=True).tokenize(text)
        list_tokenized = [word for word in list_tokenized if len(word) > 1]
        
        return list_tokenized

    def fit_transform(self, texts):
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
        wordcloud = self.wordcloud.generate(' '.join(top_words))
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.savefig(filename)
        # IMPORTANT: CLOSING FIGURE TO FREE MEMORY
        plt.close()
        
    def _define_categories(self):
        '''
        So far, hand-labelling of the latent topics
        TODO: use an ontology
        '''
        self.category = {}
        self.category['food'] = {4, 6, 7, 21, 24, 32, 34, 35, 2, 36, 25, 37,
                                38, 39, 40, 41, 42, 44, 47, 48, 53, 54, 55,
                                56, 65, 68, 69, 70, 73, 75, 79, 85, 91, 93,
                                95, 97}
        self.category['service'] = {5, 9, 15, 18, 19, 45, 58, 60, 62, 66, 71,
                                    72, 81, 90, 94, 98}
        self.category['ambience'] = {10, 16, 20, 26, 31, 61, 64}
        
        self.category['wine'] = {2, 36}
        self.category['cocktail'] = {25, 36}
        self.category['steak'] = {21}
        self.category['Chinese'] = {24}
        self.category['French'] = {47, 87}
        self.category['cheese'] = {34, 97}
        self.category['dessert'] = {35, 40, 95}
        self.category['vegetables'] = {37}
        self.category['meat'] = {42, 44, 47, 49, 51, 53, 69, 70, 85, 91}
        self.category['pork'] = {51}
        self.category['steak'] = {49, 53, 69, 70, 85}
        self.category['egg'] = {44}
        self.category['potato'] = {44, 48}
        self.category['entree'] = {38, 39}
        self.category['layout'] = {16, 26}
        self.category['noise'] = {17, 64, 82}
        self.category['music'] = {64, 82}
        self.category['location'] = {26, 50, 52, 76, 77}
        self.category['vegetarian'] = {56, 87}
        self.category['salad'] = {87}
        self.category['brunch'] = {65, 90}
        self.category['Mediterranean'] = {73}
        self.category['Indian'] = {79}
        
        self.category['excellent'] = {3, 5, 18, 20, 25, 27, 29, 33, 34, 96, 99}
        self.category['positive sentiment'] = {9, 10, 11, 12, 15, 19, 
                                               22, 28, 45, 46, 54, 59, 60, 62,
                                               63, 66, 68, 80, 81, 86, 90, 94}
        self.category['negative sentiment'] = {46, 58, 71, 94}
        self.category['experience'] = {8, 78, 92}
        self.category['positive recommendation'] = {13, 23, 30, 74, 83}
        self.category['special occasion'] = {14, 31, 43, 59, 74, 84, 89}
        self.category['reservation'] = {60}
        self.category['price'] = {67}
        self.category['cook'] = {68, 75}

        # print 'Categories created'

    def extract_onecat_top(self, texts, category, filename, top_n=15):
        '''
        INPUT: list of strings, string, integer
        OUTPUT: list of strings

        This method transforms a test set using the trained model
        to extract the top words in the latent topics corresponding
        to one category.
        It exports these words as word clouds and returns them.
        '''

        if not self.category:
            self._define_categories()
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
        
        self.cloud_fig(top_words, '../data/%s.png' % filename)

    def sentiment_one_cat(self, sentWords, texts, category):
        if not self.category:
            self._define_categories()

        texts = [sent for item in [sent_tokenize(text) for text in texts] for
                sent in item]
        V = self.vectorizer.transform(texts)
        W = self.factorizor.transform(V)

        p_pos = 0.
        p_neg = 0.
        cnt = 0
        texts = np.array(texts)
        for topic in self.category[category]:
            docs = texts[W[:, topic] > 0]

            for sent in docs:
                cnt += 1
                # Blob implementation too slow (infinitely slow)
                # bsa = BlobSentimentAnalysis(sent)
                # sentiment = bsa.sentiment()
                avg_sa = AvgSentimentAnalysis(sentWords, sent)
                sentiment = avg_sa.sentiment()
                p_pos += sentiment[0]
                p_neg += sentiment[1]

        p_pos = p_pos / float(cnt)
        p_neg = p_neg / float(cnt)

        return p_pos, p_neg


