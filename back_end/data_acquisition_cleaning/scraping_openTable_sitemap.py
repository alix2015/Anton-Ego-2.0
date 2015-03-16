import pandas as pd
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import re
import itertools
import time


'''
This file defines the functions necessary to scrape Open Table
restaurant review pages.
'''


def get_resto_links(n):
    '''
    INPUT: integer
    OUTPUT: list of strings

    Scrapes page n of Open Table index to obtain urls to the restaurant
    page on the site.
    '''
    req = 'http://www.opentable.com/opentable-sitemap.aspx?pt=100&page=%d' % n
    response = requests.get(req)
    if response.status_code != 200:
        print 'WARNING', response.status_code
    else:
        raw = response.text
        soup = BeautifulSoup(raw, 'html.parser')
        listings = soup.select('li.url-list a')
        urls = [x['href'] for x in listings]

        return urls


# The returned urls are not the one that can used to explore the review pages
# Informations can be found in the span.ReviewsDataStore options

def get_resto_data(url):
    '''
    INPUT: string
    OUTPUT: 5-tuple of strings

    Given a the index url of a restaurant, returns the profile url,
    restaurant id, restaurant name, review count, and page count.
    '''
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print 'WARNING', response.status_code
        else:
            raw = response.text
            soup = BeautifulSoup(raw, 'html.parser')
            data = soup.find('span', id='ReviewsDataStore')
            if data:
                profile_url = data['data-restprofilenlurl']
                try:
                    review_cnt = data['data-writtenreviewcount']
                except KeyError:
                    review_cnt = 0
                page_cnt = data['data-totalpages']
                rid = data['data-rid']
                rest_name = data['data-restaurantname']

                return profile_url, rid, rest_name, review_cnt, page_cnt
    except requests.exceptions.RequestException:
        time.sleep(5)
        pb.insert({'url': url})


def scrape_reviews(profile_url, rid, rest_name, review_cnt, page_cnt,
                   collection, pb):
    '''
    INPUT: string, string, string, string, string, MongoClient collection,
    MongoClient collection
    OUTPUT: None

    Scrapes raw html of review pages and store it in collection. Tracks
    failed scraping in pb.
    '''
    print 'Scraping %s' % rest_name

    if review_cnt:
        for p in xrange(int(page_cnt)):
            if not (p % 5):
                print 'Review page %d' % (p + 1)
            url = profile_url + '?rid=' + rid + '&tab=2&page=' + str(p + 1)
            try:
                response = requests.get(url)

                if response.status_code == 200:
                    collection.insert({'url': response.url,
                                       'rest_name': rest_name,
                                       'rid': rid,
                                       'html': response.content})
                time.sleep(0.5)
            except requests.exceptions.RequestException:
                time.sleep(5)
                pb.insert({'url': profile_url})


if __name__ == '__main__':
    client = MongoClient()
    db = client.opentable
    coll = db.review3
    pb = db.problem

    # 22 such pages (index of all the restaurants split into 22 pages)
    # First three scraped in the following.
    for i in xrange(1, 3):
        'Obtaining restaurants links from page %d' % i
        urls = get_resto_links(i)
        time.sleep(0.1)

        count = 0
        for url in urls:
            if not (count % 10):
                print 'Scraped %d restaurants' % count
            tup = get_resto_data(url)
            profile_url, rid, rest_name, review_cnt, page_cnt = tup

            scrape_reviews(profile_url,
                           rid,
                           rest_name,
                           review_cnt,
                           page_cnt,
                           coll)
            count += 1
            time.sleep(1)

    count = 0
    for url in urls:
        if not (count % 10):
            print '###############################'
            print 'Scraped %d restaurants' % count
        tup = get_resto_data(url)
        if tup:
            profile_url, rid, rest_name, review_cnt, page_cnt = tup

            scrape_reviews(profile_url,
                           rid,
                           rest_name,
                           review_cnt,
                           page_cnt,
                           coll,
                           pb)
            count += 1
        else:
            pb.insert({'url': url})

        time.sleep(1)
