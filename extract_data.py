import numpy as np 
import pandas as pd 
from bs4 import BeautifulSoup
from pymongo import MongoClient
import re
import itertools


class extract_data(object):
    def __init__(self):
        self.client = MongoClient()
        self.db = self.client.opentable
        self.collection = self.db.reviews

    def to_dataframe(self, filename):
        '''
        INPUT: MongoDB collection
        OUTPUT: pandas dataframe
        Extracting the reviews from the raw html stored in MongoDB collection.
        Necessary to get all the lines corresponding to a rest_name (one per page)
        '''
        # Replaced the direct importation from MongoDB by an importation
        # through JSON
        # cursor = collection.find({}, {'rest_name': 1, 'html': 1, '_id': 0})
        # df = pd.DataFrame(list(cursor))

        # List of restaurants
        restos = list(self.collection.find({}, {'rest_name':1, '_id':0}))
        restos = [d['rest_name'] for d in restos[:10]]
        restos = set(restos)

        df_list = []
        for r in restos:
            cursor = self.collection.find({'rest_name': r}, 
                                          {'rest_name': 1, 'html': 1, '_id': 0})
            df = pd.DataFrame(list(cursor))
            df['data'] = df['html'].map(self.data)

            # print 'df shape is ', df.shape
            # print df.head(1)

            df2 = pd.concat([pd.Series(row['rest_name'], row['data']) for 
                            _, row in df.iterrows()]).reset_index()
            # print 'df2.shape is ', df2.shape
            # print df2.head(1)
            # df2.columns = ['rest_name', 'data']

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

            # print 'df3 shape is ', df3.shape

            df_list.append(df3)

        df3 = pd.concat(df_list)
        df3.to_pickle(filename)

        return df3


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
        # print review_lengths
        # print 'review_lengths length is %d' % len(review_lengths)

        # Title of the reviews
        listings = soup.select('h4.review-title')
        review_titles = [x.text.strip() for x in listings]
        # print 'review_titles length is %d' % len(review_titles)

        # Ratings
        listings = soup.find_all('meta', {'itemprop': 'ratingValue'})
        ratings = [float(x['content']) for x in listings]
        # print 'ratings length is %d' % len(ratings)

        # Category ratings
        listings = soup.select('span.review-stars-results-num')
        detailed_ratings = [float(x.text.strip()) for x in listings]
        n = len(detailed_ratings) / 3
        food_rating = [detailed_ratings[3*i] for i in xrange(n)]
        service_rating = [detailed_ratings[3*i + 1] for i in xrange(n)]
        ambience_rating = [detailed_ratings[3*i + 2] for i  in xrange(n)]

        # print 'food_rating length is %d' % len(food_rating)
        # print 'service_rating length is %d' % len(food_rating)
        # print 'ambience_rating lenght is %d' % len(ambience_rating)
        
        return [t for t in itertools.izip(review_titles,
                                          review_lengths,
                                          reviews,
                                          ratings,
                                          food_rating,
                                          service_rating,
                                          ambience_rating)]


if __name__ == '__main__':
    filename = '../data/reviews_SF.pkl'
    
    ed = extract_data()

    df = ed.to_dataframe(filename)

    print df.shape
    print 'Info:'
    print df.info()

