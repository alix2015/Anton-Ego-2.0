import numpy as np 
import pandas as pd 
from categories import Categories
from processing_data import TopicExtraction
from sentiment_analysis import BlobSentimentAnalysis
import dill
import timeit


# Utilities for a pipeline based on dataframes

def build_model(df, n_topics, ngram_range, max_words, max_iter, model_filename,
                top_filename=None):
    '''
    INPUT: pandas dataframe, integer, integer 2-tuple, integer, integer, string
    [string]
    OUTPUT: TopicExtraction object

    This functions uses the data in df to initialize a TopicExtraction
    '''
    
    rest_names = df['rest_name'].dropna().unique().tolist()
    print 'Number of restaurants: %d' % len(rest_names)
    te = TopicExtraction(rest_names=rest_names, n_topics=n_topics, sentence=True,
                         ngram_range=ngram_range, max_words=max_words,
                         max_iter=max_iter)


    if top_filename:
        top_words = te.extract_top_words(df['review'], top_n=15,
                                         top_filename=top_filename)

    else:
        te.fit_transform(df['reviews'])

    if model_filename:
        with open(model_filename, 'w') as f:
            dill.dump(te, f)
            print 'Finished pickling the model'
    return te


def model_initializing(data_file, model_file, verbose=True):
    '''
    INPUT: string, string, [string, boolean]
    OUTPUT: None

    This function is called to initialize a TopicExtraction model
    using the data loaded from data_file pickle file, outputting it
    in model_filename out. The optional boolean input switches the verbosity.
    '''
    
    df = pd.read_pickle(data_file)
    texts = df['review']

    n_topics = 100
    ngram_range = (2, 2)
    max_words = 5000
    max_iter = 400

    top_filename = '../../data/te_2_20_%d_%dgram_max_%d_100_extraStopW.txt' % \
                    (n_topics, ngram_range[1], max_words)

    tic = timeit.default_timer()
    te = build_model(df, n_topics, ngram_range, max_words, max_iter,
                     model_filename, top_filename)
    toc = timeit.default_timer()
    if verbose:
        print 'Building model in %3.f seconds' % (toc - tic)


def build_results(rest_name, base, base_fig=None, verbose=True, export=False):
    '''
    INPUT: string, string, string, [boolean, boolean]
    OUTPUT: dictionary, dictionary

    This function uses an initialized TopicExtraction model
    and a dataset to perform latent feature extraction and
    sentiment analysis on this dataset. It returns a dictionary
    of sentences inedxed by category and a dictionary of sentiment
    output indexed by category.
    The optional verbose boolean controls verbosity.
    The optional export boolean controls whether sentence categorization
    and sentiments are pickled.
    '''
    # paths have to be written from the front_end/app/ viewpoint
    df = pd.read_pickle(base + 'df_clean2a.pkl')
    texts = df['review']

    model_filename = base + 'te_2a_extraSW.pkl'
    te = dill.load(open(model_filename, 'rb'))
    categories = Categories()
    special = {'food', 'service', 'ambience'}

    sent_analyzer = BlobSentimentAnalysis()

    texts = df[df['rest_name'] == rest_name]['review'].values
    rid = df[df['rest_name'] == rest_name]['rid'].unique()[0]

    if verbose:
        print 'Running top_cat...'
    tic = timeit.default_timer()
    top_cat = te.top_categories(texts, cat=categories)
    for c in special:
        if c not in top_cat:
            top_cat = np.append(top_cat, c)
    if verbose:
        print top_cat
    tac = timeit.default_timer()
    if verbose:
        print 'Finished top_cat in %.3f seconds' % (tac - tic)

    sentences = {}
    sentiments = {}
    if verbose:
        print 'Looping over categories...'
    tic = timeit.default_timer()
    for c in top_cat:
        if base_fig:
            cloud_name = '%s_%s' % (rid, c)
            cloud_name = base_fig + cloud_name
            print cloud_name
            te.extract_onecat_topwords(texts, c, cloud_name, base_fig)
        else:
            te.extract_onecat_topwords(texts, c)
        sentences[c] = te.extract_onecat_sentences(texts, c, token=False)
        sentiments[c] = sent_analyzer.sentiment_sentences(sentences[c])
    tac = timeit.default_timer()
    if verbose:
        print 'End looping in %.3f seconds' % (tac - tic)

    if export:
        filename = '%s_snippets.pkl' % rid
        filename = base + filename
        tic = timeit.default_timer()
        with open(filename, 'w') as f:
            dill.dump(sentences, f)
        f.close()
        tac = timeit.default_timer()
        if verbose:
            print 'Finished pickling sentences in %.3f seconds' % (tac - tic)

        filename = '%s_sentiments.pkl' % rid
        filename = base + filename
        tic = timeit.default_timer()
        with open(filename, 'w') as f:
            dill.dump(sentiments, f)
        f.close()
        tac = timeit.default_timer()
        if verbose:
            print 'Finished pickling sentiments in %.3f seconds' % (tac - tic) 

        top_cat = [item for item in top_cat if item not in {'food', 'service', 
                   'ambience'}]    

    return sentences, sentiments

def example_from_backend():
    '''
    INPUT: None
    OUTPUT: None

    This is an example of how to call latent topic extraction
    and sentiment analysis from the back_end folder.
    '''
    # rest_name = 'Il Borgo'
    base = '../front_end/data/'
    base_fig = '../front_end/app/static/img/'
    df = pd.read_pickle(base + 'df_clean2a.pkl')
    rest_names = df['rest_name'].unique()

    for rest_name in rest_names:
        rid = int(df[df['rest_name'] == rest_name]['rid'].unique()[0])
        if rid not in calculated_rid:
            calculated_rid.add(rid)
            print calculated_rid
            print len(calculated_rid)
            sentences, sentiments = build_results(rest_name, base, base_fig)

            for cat, sent in sentences.iteritems():
                print cat
                print sent

            for cat, sent in sentiments.iteritems():
                print cat
                print sent 


if __name__ == '__main__':
    # data_file = '../front_end/data/df_clean2a.pkl'
    # model_filename = '../front_end/data/te_2a_extraSW.pkl'

    example_from_backend()
