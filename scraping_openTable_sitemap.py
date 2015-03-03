import pandas as pd 
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import re
import itertools
import time


# 22 such pages (index of all the restaurants split into 22 pages)
def get_resto_links(n):
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


def scrape_reviews(profile_url,
                   rid,
                   rest_name,
                   review_cnt,
                   page_cnt,
                   collection,
                   pb):
    print 'Scraping %s' % rest_name
    
    if review_cnt:
        for p in xrange(int(page_cnt)):
            if not (p % 5):
                print 'Review page %d' % (p + 1)
            url = profile_url + '?rid=' + rid + '&tab=2&page=' + str(p + 1)
            # print url
            try:
                response = requests.get(url) 
                # print response.status_code
                
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
    coll = db.review
    pb = db.problem

# url = 'http://www.opentable.com/rest_profile_reviews.aspx?rid=99991&tab=2'
    # for i in xrange(1, 5):
    #     'Obtaining restaurants links from page %d' % i
    #     urls = get_resto_links(i)
    #     time.sleep(0.1)

    #     count = 0
    #     for url in urls:
    #         if not (count % 10):
    #             print 'Scraped %d restaurants' % count
    #         tup = get_resto_data(url)
    #         profile_url, rid, rest_name, review_cnt, page_cnt = tup

    #         scrape_reviews(profile_url,
    #                        rid,
    #                        rest_name, 
    #                        review_cnt, 
    #                        page_cnt, 
    #                        coll)
    #         count += 1
    #         time.sleep(1)
    'Obtaining restaurants links from page %d' % 1
    urls = get_resto_links(1)
    time.sleep(0.1)

    count = 0
    for url in urls[2768:]:
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
