import numpy as np 
import pandas as pd 
from bs4 import BeautifulSoup
from pymongo import MongoClient
import re
import itertools
import timeit


class ExtractData(object):
    def __init__(self):
        self.client = MongoClient()
        self.db = self.client.opentable
        # TODO: not harcoding the name of the collection
        self.collection = self.db.review2
        self.collectionOut = self.db.clean2

    def to_dataframe(self, filename):
        '''
        INPUT: ExtractData object, string
        OUTPUT: pandas dataframe
        Extracting the reviews from the raw html stored in MongoDB self.collection.
        Necessary to get all the lines corresponding to a rest_name (one per page).
        Pickle the dataframe to filename
        '''
        
        # List of restaurants
        restos = self.collection.distinct('rest_name')

        df_list = []
        for r in restos:
            cursor = self.collection.find({'rest_name': r}, 
                                          {'rest_name': 1, 'html': 1, '_id': 0})
            df = pd.DataFrame(list(cursor))
            df['data'] = df['html'].map(self.data)

            df2 = pd.concat([pd.Series(row['rest_name'], row['data']) for 
                            _, row in df.iterrows()]).reset_index()
            
            df3 = df2['index'].apply(pd.Series)
            df3['rest_name'] = df['rest_name']
            df3.columns = ['review_titles',
                           'review_lengths',
                           'reviews',
                           'ratings',
                           'food_rating',
                           'service_rating',
                           'ambience_rating',
                           'rest_name']

            df_list.append(df3)

        df3 = pd.concat(df_list)
        # Getting rid of duplicates: the review field is the most defining one
        df3 = df3.drop_duplicates('reviews')
        df3.to_pickle(filename)

        return df3

    def to_mongo(self):
        '''
        INPUT: ExtractData object
        OUTPUT: None
        Process the raw reviews in self.collection and store them in
        self.collectionOut.
        '''

        # # List of restaurants
        # restos = self.collection.distinct('rest_name')
        # Cursor to process all the documents
        cursor = self.collection.find({})

        for doc in cursor:
            url = doc['url']
            rest_name = doc['rest_name']
            rid = doc['rid']
            html = doc['html']
            data = self.data(html)

            for tup in data:
                self.collectionOut.insert({'url': url,
                                          'rest_name': rest_name,
                                          'rid': rid,
                                          'review_title': tup[0],
                                          'review_length': tup[1],
                                          'review': tup[2],
                                          'rating': tup[3],
                                          'food_rating': tup[4],
                                          'service_rating': tup[5],
                                          'ambience_rating': tup[6]})



    def data(self, raw):
        '''
        INPUT: raw html
        OUTPUT: list of parameters extracted from html
        '''
        soup = BeautifulSoup(raw, 'html.parser')
        # Body of the reviews
        listings = soup.select('div.review-content')
        reviews = [x.text.strip() for x in listings]
        review_lengths = [len(r) for r in reviews]
        
        # Title of the reviews
        listings = soup.select('h4.review-title')
        review_titles = [x.text.strip() for x in listings]
        
        # Ratings
        listings = soup.find_all('meta', {'itemprop': 'ratingValue'})
        ratings = [float(x['content']) for x in listings]
        
        # Category ratings
        listings = soup.select('span.review-stars-results-num')
        detailed_ratings = [float(x.text.strip()) for x in listings]
        n = len(detailed_ratings) / 3
        food_rating = [detailed_ratings[3*i] for i in xrange(n)]
        service_rating = [detailed_ratings[3*i + 1] for i in xrange(n)]
        ambience_rating = [detailed_ratings[3*i + 2] for i  in xrange(n)]

        return [t for t in itertools.izip(review_titles,
                                          review_lengths,
                                          reviews,
                                          ratings,
                                          food_rating,
                                          service_rating,
                                          ambience_rating)]


def main1():
    filename = '../data/reviews_SF.pkl'
    
    ed = ExtractData()

    df = ed.to_dataframe(filename)

    print df.shape
    print 'Info:'
    print df.info()


def main2():
    extractor = ExtractData()
    tic = timeit.default_timer()
    extractor.to_mongo()
    toc = timeit.default_timer()
    print 'Extraction finished in %.3f seconds.' % (toc - tic)


if __name__ == '__main__':
    main2()