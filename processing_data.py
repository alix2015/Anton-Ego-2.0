import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, accuracy_score
from wordcloud import WordCloud
# from sklearn.pipeline import Pipeline
# import cPickle as pickle


class EDA(object):
    def __init__(self, verbose=True, figsize=(10, 8)):
        self.verbose = verbose
        self.figsize = figsize

    def review_size_distribution(self, df, filename):
        '''
        INPUT: pandas dataframe
        OUTPUT: none
        Plot the review size distribution for the entire corpus
        '''
        plt.figure(figsize=self.figsize)
        df['review_lengths'].hist(bins=100)
        plt.xlabel('Review length')
        plt.ylabel('Number of reviews')
        plt.title('Review length distribution')
        plt.savefig(filename + '.png')

        if self.verbose:
            print 'Review length statistics'
            print df['review_lengths'].describe()

    def ratings_distribution(self, df, filename):
        plt.figure(figsize=self.figsize)
        df['ratings'].hist(bins=20)
        plt.xlabel('Rating')
        plt.ylabel('Number of reviews')
        plt.title('Overall rating distribution')
        plt.savefig(filename + '_overall.png')

        plt.figure(figsize=self.figsize)
        df['food_rating'].hist(bins=20)
        plt.xlabel('Rating')
        plt.ylabel('Number of reviews')
        plt.title('Food rating distribution')
        plt.savefig(filename + '_food.png')

        plt.figure(figsize=self.figsize)
        df['service_rating'].hist(bins=20)
        plt.xlabel('Rating')
        plt.ylabel('Number of reviews')
        plt.title('Service rating distribution')
        plt.savefig(filename + '_service.png')

        plt.figure(figsize=self.figsize)
        df['ambience_rating'].hist(bins=20)
        plt.xlabel('Rating')
        plt.ylabel('Number of reviews')
        plt.title('Ambience rating distribution')
        plt.savefig(filename + '_ambience.png')

        if self.verbose:
            print 'Ratings statistics:'
            print 'Overall'
            print df['ratings'].describe()
            print '-------------------'
            print 'Food'
            print df['food_rating'].describe()
            print '-------------------'
            print 'Service'
            print df['service_rating'].describe()
            print '-------------------'
            print 'Ambience'
            print df['ambience_rating'].describe()


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
        self.H_ = None

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
        
        return W

    def top_words(self, texts, top_n=10):
        self.fit_transform(texts)
        top_words = {}
        for topic in xrange(self.n_topics):
            top_words_idx = np.argsort(self.H_[topic, :])[-1:-(top_n + 1):-1]
            top_words[topic] = [self.vectorizer.get_feature_names()[i] for
                                i in top_words_idx]
        
        return top_words

    def extract_nouns(self, sentence):
        '''
        Only keep nouns for each line
        '''
        # Getting rid of non_ascii characters
        text = nltk.word_tokenize(re.sub(r'[^\x00-\x7F]+', ' ', sentence))
        word_tags = nltk.pos_tag(text)
        
        return ' '.join([word_tag[0] for 
                        word_tag in word_tags if 
                        word_tag[1][:2] == 'NN'])



if __name__ == '__main__':
    data_SF = '../data/reviews_SF.pkl'
    data_1 = '../data/reviews_1.pkl'
    data_2 = '../data/reviews_2.pkl'
    # length_distribution_filename = '../data/length_distribution.png'

    df_SF = pd.read_pickle(data_SF)
    df1 = pd.read_pickle(data_1)
    df2 = pd.read_pickle(data_2)

    df = pd.concat([df_SF, df1, df2])
    df = df.drop_duplicates('reviews')

    # length_distribution_filename = '../data/length_distribution'
    # ratings_filename = '../data/ratings_distribution'
    # eda = EDA()
    # eda.review_size_distribution(df, length_distribution_filename)
    # eda.ratings_distribution(df, ratings_filename)
    
    # Getting rid of short reviews (< 100 characters)
    # df = df[df['review_lengths'] > 100]   
    
    n_topics = 100
    ngram_range = (2, 2)
    max_words = 5000
    max_iter = 400

    # top_filename = '../data/topics_%d_%dgram_max_%d_lg_d.txt' % (n_topics,
    #                                                              ngram_range[1],
    #                                                              max_words)

    # te = TopicExtraction(n_topics=n_topics,
    #                      ngram_range=ngram_range,
    #                      max_words=max_words,
    #                      max_iter=max_iter)

    top_filename = '../data/topics_%d_%dgram_max_%d_lg_s.txt' % (n_topics,
                                                                 ngram_range[1],
                                                                 max_words)

    te = TopicExtraction(n_topics=n_topics,
                         sentence=True,
                         ngram_range=ngram_range,
                         max_words=max_words,
                         max_iter=max_iter)

    top_words = te.top_words(df['reviews'], top_n=15)

    # with open(top_filename, 'w') as f:
    #     f.write('n_gram: %d, %d' % (ngram_range[0], ngram_range[1]))
    #     f.write('\n')
    #     f.write('n_topics: %d' % n_topics)
    #     f.write('\n')
    #     f.write('-------------------------')
    #     f.write('\n')
    #     for topic in top_words:
    #         f.write('Topic %d' % topic)
    #         f.write('\n')
    #         for word in top_words[topic]:
    #             f.write(word)
    #             f.write('\n')
    #         f.write('-------------------------')
    #         f.write('\n')
    # f.close()
    # TODO: add path to font file
    path = '/Library/Fonts/Georgia.ttf'
    wordcloud = WordCloud(font_path=path).generate(' '.join(top_words[0]))
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('../data/wordcloud_0.png')
