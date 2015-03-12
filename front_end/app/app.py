from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from itertools import izip
# from pymongo import MongoClient

# client = MongoClient()
# coll = client.opentable.clean2

df = pd.read_pickle('../data/df_clean2.pkl')

rest_names = df['rest_name'].unique()
rid = df['rid'].unique()
restos = [t for t in izip(rid, rest_names)]

PORT = 8888
app = Flask(__name__)


# OUR HOME PAGE
#============================================
@app.route('/')
def home():
    return render_template('home.html')
    

# QUERY PAGE
#============================================
@app.route('/submit_page')
def submission_page():
    return render_template('submit2.html',
                           restos=restos)

# RESULT PAGE
#============================================
@app.route('/topic', methods=['POST', 'GET'])
def predict_page():
    # get data from request form, the key is the name you set in your form
    # TODO: 'Sorry, the Critic has not yet visited this restaurant'
    rid = request.form['user_choice']
    print rid
    # cursor = coll.find({'rest_name': rest_name}, {'url': 1, '_id': 0})
    # rest_link = cursor['url']
    # cursor = coll.find({'rest_name': rest_name}, {'reviews': 1, '_id': 0})
    # top1, top2, top3, top4, top5 = te.top_categories(reviews)
    # mongo query for ratings
    # rating, food, service, ambience = 
    topA = 'steak'
    topB = 'birthday'
    topC = 'wine'
    topD = 'location'
    topE = 'date'
    mask = df['rid'] == rid
    rest_name = df[mask]['rest_name'].values[0]
    rest_link = df[mask]['url'].values[0]
    rating = '%.2f' % df[mask]['rating'].mean()
    rating_food = '%.2f' %  df[mask]['food_rating'].mean()
    rating_service = '%.2f' %  df[mask]['service_rating'].mean()
    rating_ambience = '%.2f' % df[mask]['ambience_rating'].mean()
    return render_template('result2a.html', rest_name=rest_name,
                           rest_names=rest_names, restos=restos,
                           rest_link=rest_link, topA=topA, topB=topB, topC=topC,
                           topD=topD, topE=topE, rating=rating,
                           rating_food=rating_food, rating_service=rating_service,
                           rating_ambience=rating_ambience)
    
    
@app.route('/topic/detail', methods=['POST'])
def detail():
    rest_name = todo #request.form['user_input']
    category = request.a['category']
    # rest_link = coll.find({'rest_name': rest_name}, {'url': 1, '_id': 0})
    rest_link = 'http://www.opentable.com/lardoise'
    return render_template('result_detail.html', rest_name=rest_name,
                           rest_link=rest_link, category=category)

# INFO PAGE
#============================================
@app.route('/info')
def info_page():
    return render_template('info2.html')

# INFO PAGE
#============================================
@app.route('/contact')
def contact_page():
    return render_template('contact2.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)