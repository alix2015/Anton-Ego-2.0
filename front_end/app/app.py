from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from itertools import izip
import sys
sys.path.insert(0, '../../back_end/')
from main_df import build_results

base = '../data/'
base_fig = 'static/img/'

df = pd.read_pickle(base + 'df_clean2a.pkl')
rest_names = df['rest_name'].unique()
rid = df['rid'].unique()
restos = [t for t in izip(rid, rest_names)]
restos.sort(key=lambda t: t[1])

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
    # get data from request form, the key is the restaurant id
    rid = request.form['user_choice']
    
    mask = df['rid'] == rid
    rest_name = df[mask]['rest_name'].values[0]
    rest_link = df[mask]['url'].values[0]
    rating = '%.2f' % df[mask]['rating'].mean()
    rating_food = '%.2f' %  df[mask]['food_rating'].mean()
    rating_service = '%.2f' %  df[mask]['service_rating'].mean()
    rating_ambience = '%.2f' % df[mask]['ambience_rating'].mean()

    sentences, sentiments = build_results(rest_name, base, base_fig)
    for tup in sentiments:
        print tup
    special = {'food', 'service', 'ambience'}
    top = [cat for cat in sentences.keys() if cat not in special]
    cloud_food = base_fig + rid + '_food.png'
    cloud_service = base_fig + rid + '_service.png'
    cloud_ambience = base_fig + rid + '_ambience.png'
    clouds = [base_fig + rid + '_' + cat + '.png' for cat in top]

    return render_template('result2a.html', rest_name=rest_name,
                           rest_names=rest_names, restos=restos,
                           rest_link=rest_link, top=top,
                           rating=rating, sentiments=sentiments,
                           sentences=sentences, rating_food=rating_food,
                           rating_service=rating_service,
                           rating_ambience=rating_ambience,
                           cloud_food=cloud_food, cloud_service=cloud_service,
                           cloud_ambience=cloud_ambience, clouds=clouds)
    
    
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