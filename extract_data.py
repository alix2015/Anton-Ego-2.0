import numpy as np 
import pandas as pd 
from bs4 import BeautifulSoup
from pymongo import MongoClient
import re
import itertools



def to_dataframe(collection, filename):
    '''
    INPUT: MongoDB collection
    OUTPUT: pandas dataframe
    Extracting the reviews from the raw html stored in MongoDB collection.
    Necessary to get all the lines corresponding to a rest_name (one per page)
    '''
    cursor = collection.find({}, {'rest_name': 1, 'html': 1, '_id': 0})
    df = pd.DataFrame(list(cursor))

    df['data'] = df['html'].map(extract_data)

    print 'df shape is ', df.shape

    df2 = pd.concat([pd.Series(row['rest_name'], row['data']) for 
                    _, row in df.iterrows()]).reset_index()
    print 'df2.shape is ', df2.shape

    df3 = df2['data'].apply(pd.Series)
    df3['rest_name'] = df['rest_name']

    print 'df3 shape is ', df3.shape

    df3.to_pickle(filename)

    return df2


def extract_data(raw):
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


if __name__ == '__main__':
    client = MongoClient()
    db = client.opentable
    coll = db.reviews

    filename = '../data/reviews.pkl'

    df = to_dataframe(coll, filename)
    print df.shape
    print 'Info:'
    print df.info()

