import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from categories import Categories
from topics import TopicExtraction
from sentiment_analysis import BlobSentimentAnalysis
import dill
import timeit


# -----------------------------------------------------------------------------
# UTILITIES TO USE PANDAS DATAFRAME
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# MODEL INITIALIZATION
# -----------------------------------------------------------------------------

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
    te = TopicExtraction(rest_names=rest_names, n_topics=n_topics,
                         sentence=True, ngram_range=ngram_range,
                         max_words=max_words, max_iter=max_iter)

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

    df = pd.read_csv(data_file)
    texts = df['review']

    n_topics = 100
    ngram_range = (2, 2)
    max_words = 5000
    max_iter = 400

    top_filename = '../../data/te_3a_20_%d_%dgram_max_%d_100_extraStopW.txt' %\
                   (n_topics, ngram_range[1], max_words)

    tic = timeit.default_timer()
    te = build_model(df, n_topics, ngram_range, max_words, max_iter,
                     model_filename, top_filename)
    toc = timeit.default_timer()
    if verbose:
        print 'Building model in %3.f seconds' % (toc - tic)

# -----------------------------------------------------------------------------
# SENTIMENT DISTRIBUTION
# -----------------------------------------------------------------------------


def build_results2(rest_name, base, base_fig=None, verbose=True, export=False):
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
    df = pd.read_csv(base + 'df_clean3a.csv')
    # df = pd.read_pickle(base + 'df_clean2a.pkl')
    texts = df['review']

    model_filename = base + 'te_3a_extraSW.pkl'
    te = dill.load(open(model_filename, 'rb'))
    categories = Categories()
    special = {'Food', 'Service', 'Ambience'}

    sent_analyzer = BlobSentimentAnalysis()

    texts = df[df['rest_name'] == rest_name]['review'].values
    rid = df[df['rest_name'] == rest_name]['rid'].unique()[0]

    if verbose:
        print 'Running top_cat...'
    tic = timeit.default_timer()
    top_cat = te.top_categories(texts, cat=categories)
    for c in special:
        if c not in top_cat:
            top_cat.append(c)
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
        # if base_fig:
        #     cloud_name = '%d_%s' % (rid, c)
        #     cloud_name = base_fig + cloud_name
        #     print cloud_name
        #     te.extract_onecat_topwords(texts, c, cloud_name, base_fig)
        # else:
        #     te.extract_onecat_topwords(texts, c)
        sentences[c] = te.extract_onecat_sentences(texts, c, token=False)
        sentiments[c] = sent_analyzer.sentiment_sentences(sentences[c])

        if base_fig:
            for c in sentiments:
                plt.figure(figsize=(10, 8))
                sent = np.array([s[0][0] for s in sentiments[c]])
                if sent.shape[0] > 1:
                    plt.hist(sent, bins=20)
                    plt.savefig(base_fig + ('%d_%s_sentiments.png' % (rid, c)))
                    plt.figure(figsize=(10, 8))
                    subj = np.array([s[0][1] for s in sentiments[c]])
                    plt.hist(subj, bins=20)
                    plt.savefig(base_fig + ('%d_%s_subjsectivity.png' % (rid, c)))
                    plt.close()
    tac = timeit.default_timer()
    if verbose:
        print 'End looping in %.3f seconds' % (tac - tic)

    if export:
        tic = timeit.default_timer()
        filename = base + ('%d_sentiments.csv' % rid)
        sent = pd.DataFrame([[c, t[0][0], t[0][1], t[1]] for c, v in
                            sentiments.iteritems() for t in v])
        sent.columns = ['category', 'sentiment', 'subjectivity', 'sentence']
        sent.to_csv(filename)
        tac = timeit.default_timer()
        if verbose:
            print 'Finished exporting sentiments in %.3f seconds' % (tac - tic)

        top_cat = [item for item in top_cat if item not in {'Food', 'Service',
                   'Ambience'}]

    return sentiments

# -----------------------------------------------------------------------------
# EXAMPLE OF USE
# -----------------------------------------------------------------------------


def example_from_backend():
    '''
    INPUT: None
    OUTPUT: None

    This is an example of how to call latent topic extraction
    and sentiment analysis from the back_end folder.
    '''
    base = '../front_end/data/'
    base_fig = '../front_end/app/static/img/'
    df = pd.read_csv(base + 'df_clean3a.csv')

    print df.shape

    rest_names = df['rest_name'].unique()

    calculated_rid = set([])
    cnt = 0

    for rest_name in rest_names:
        rid = df[df['rest_name'] == rest_name]['rid'].unique()[0]
        if cnt > 5:
            break
        if rid not in calculated_rid:
            cnt += 1
            calculated_rid.add(rid)
            print calculated_rid
            print len(calculated_rid)
            sentiments = build_results2(rest_name, base, export=True)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    data_file = '../front_end/data/df_clean3a.csv'
    model_filename = '../front_end/data/te_3a_extraSW.pkl'

    model_initializing(data_file, model_filename)

    example_from_backend()
