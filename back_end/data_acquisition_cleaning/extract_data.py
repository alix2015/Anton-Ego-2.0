import numpy as np 
import pandas as pd 
from bs4 import BeautifulSoup
from pymongo import MongoClient
import re
from itertools import izip
import timeit
import random


class ExtractData(object):
    '''
    For small datasets, dataframes can be used.
    For larger datasets, Mongo database is more adapted.
    '''
    def __init__(self):
        self.client = MongoClient()
        self.db = self.client.opentable
        # TODO: not harcoding the name of the collection
        self.collection = self.db.review2
        self.collectionOut = self.db.clean2a


    def to_mongo(self):
        '''
        INPUT: ExtractData object
        OUTPUT: None

        Process the raw reviews in MongoDB self.collection and store them in
        MongoDB self.collectionOut.
        '''

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
                                          'ambience_rating': tup[6],
                                          'address': tup[7]})

    def to_dataframe(self, filename):
        '''
        INPUT: ExtractData object, string
        OUTPUT: pandas dataframe
        
        Extracting the reviews from the raw html stored in MongoDB
        self.collection.
        Necessary to get all the lines corresponding to a rest_name 
        (one per page).
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
                           'address']

            df_list.append(df3)

        df3 = pd.concat(df_list)
        # Getting rid of duplicates: the review field is the most defining one
        df3 = df3.drop_duplicates('reviews')
        df3.to_pickle(filename)

        return df3



    def data(self, raw):
        '''
        INPUT: raw html
        OUTPUT: list of strings

        This methods outputs parameters extracted from html
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

        # Address
        listings = soup.find_all('div', {'itemprop': 'streetAddress'})
        address = [x.text.strip() for x in listings]
        
        return [t for t in itertools.izip(review_titles,
                                          review_lengths,
                                          reviews,
                                          ratings,
                                          food_rating,
                                          service_rating,
                                          ambience_rating,
                                          address)]


def parse_address(address):
    '''
    INPUT: string
    OUTPUT: 3-tuple of strings

    This function parses most address in the USA and Canada
    '''
    pattern = r'([\w+\s]+[A-Z]\w+\.*)([A-Z]\w+\s*\w*,\s[A-Z][A-Z])\s+(\w\w\w\s*\w\w+)'
    pattern = re.compile(pattern)
    m = pattern.search(address)
    if m:
        # street address, city with state, zipcode
        return m.group(1), m.group(2), m.group(3)
    else:
        return '', '', ''


def processing_tomongo():
    '''
    INPUT: None
    OUTPUT: None

    Process raw html data dumped into a Mongo DB during scraping
    into another Mongo DB of cleaned data
    '''
    extractor = ExtractData()
    tic = timeit.default_timer()
    extractor.to_mongo()
    toc = timeit.default_timer()
    print 'Extraction finished in %.3f seconds.' % (toc - tic)

def sampling_SF():
    '''
    INPUT: None
    OUTPUT: None

    Sampling restaurants in San Francisco, CA for the front end
    '''
    client = MongoClient()
    coll = client.opentable.clean2a

    address = coll.distinct('address')
    parsed_address = [parse_address(a) for a in address]
    sf_mask = [t[1] == 'San Francisco, CA' for t in parsed_address]
    sf_address = [t[0] for t in izip(address, sf_mask) if t[1]]

    sf_rest = []
    for a in sf_address:
        cursor = coll.find({'address': a}, {'rest_name': 1})
        sf_rest.append(cursor.next()['rest_name'])

    print len(sf_rest)

    # SF restaurants
    cursor = coll.find({'rest_name': {'$in': sf_rest}})
    df = pd.DataFrame(list(cursor))
    print df.shape
    df.to_pickle('../../front_end/data/df_clean2a.pkl')


if __name__ == '__main__':
    processing_tomongo()
    sampling_SF()