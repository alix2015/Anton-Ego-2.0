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
        if not self.trained:
            self.fit_transform(texts)
        top_words = {}
        for topic in xrange(self.n_topics):
            top_words_idx = np.argsort(self.H_[topic, :])[-1:-(top_n + 1):-1]
            top_words[topic] = [self.vectorizer.get_feature_names()[i] for
                                i in top_words_idx]
            if wordcloud:
                wordcloud = self.wordcloud.generate(' '.join(top_words[topic]))
                plt.figure(figsize=(10, 8))
                plt.imshow(wordcloud)
                plt.axis('off')
                plt.savefig('../data/wordcloud_%d.png' % topic)
        
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

        def extract_cat_top(self,
                            texts,
                            top_n=10,
                            )


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
                top_filename,
                model_filename):
    
    te = TopicExtraction(n_topics=n_topics,
                         sentence=True,
                         ngram_range=ngram_range,
                         max_words=max_words,
                         max_iter=max_iter)

    top_words = te.extract_top_words(df['reviews'],
                                     top_n=15)
                                     # top_filename=top_filename,
                                     # wordcloud=True)

    if model_filename:
        with open(model_filename, 'w') as f:
            dill.dump(te, f)
            print 'Finished pickling the model'
    return te



if __name__ == '__main__':
    data_SF = '../data/reviews_SF.pkl'
    data_1 = '../data/reviews_1.pkl'
    data_2 = '../data/reviews_2.pkl'

    df =  build_data([data_SF, data_1, data_2], min_rev_len=100)

    n_topics = 100
    ngram_range = (2, 2)
    max_words = 5000
    max_iter = 400

    top_filename = '../data/topics_%d_%dgram_max_%d_100_s.txt' % \
                    (n_topics, ngram_range[1], max_words)

    model_filename = '../data/topics_1.pkl'

    te = TopicExtraction(n_topics=n_topics,
                         sentence=True,
                         ngram_range=ngram_range,
                         max_words=max_words,
                         max_iter=max_iter)

    top_words = te.extract_top_words(df['reviews'],
                                     top_n=15,
                                     top_filename=top_filename,
                                     wordcloud=False)

    # Pickling does not work as such due to my_tokenize
    te = build_model(df, n_topics, ngram_range, max_words, max_iter, 
                     top_filename, model_filename)
    