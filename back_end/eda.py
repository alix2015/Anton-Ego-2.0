import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


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


def exploring(df,
              length_distribution_filename,
              ratings_filename):
    eda = EDA()
    eda.review_size_distribution(df, length_distribution_filename)
    eda.ratings_distribution(df, ratings_filename)    


def build_data(filenames, min_rev_len=0):
    df_list = []
    for file in filenames:
        df_list.append(pd.read_pickle(file))

    df = pd.concat(df_list)
    df = df.drop_duplicates('reviews')

    df = df[df['review_lengths'] > min_rev_len]

    return df

if __name__ == '__main__':
    data_SF = '../data/reviews_SF.pkl'
    data_1 = '../data/reviews_1.pkl'
    data_2 = '../data/reviews_2.pkl'
    
    length_distribution_filename = '../data/length_distribution'
    ratings_filename = '../data/ratings_distribution'
    df =  build_data([data_SF, data_1, data_2])
    
    eda = EDA()
    eda.review_size_distribution(df, length_distribution_filename)
    eda.ratings_distribution(df, ratings_filename)
