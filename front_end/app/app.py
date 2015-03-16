from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from itertools import izip
import plotly.plotly as py 
from plotly.graph_objs import *
import dill
import sys
sys.path.insert(0, '../../back_end/')
from main_df import build_results2

base = '../data/'
base_fig = 'static/img/'

df = pd.read_csv(base + 'df_clean3a.csv')
rid = df['rid'].astype(str).unique()
rest_names = [df[df['rid'] == int(r)]['rest_name'].unique()[0] for r in rid]
restos = [t for t in izip(rid, rest_names)]
restos.sort(key=lambda t: t[1])

PORT = 8888
# PORT = 80  # (AWS)
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
    
    filename = base + rid + '_sentiments.csv'
    print filename
    
    mask = df['rid'] == int(rid)
    rest_name = df[mask]['rest_name'].values[0]
    rest_link = df[mask]['url'].values[0]
    rating = '%.2f' % df[mask]['rating'].mean()
    rating_food = '%.2f' %  df[mask]['food_rating'].mean()
    rating_service = '%.2f' %  df[mask]['service_rating'].mean()
    rating_ambience = '%.2f' % df[mask]['ambience_rating'].mean()

    sentiments = build_results2(rest_name, base, verbose=False)
    plot_url = {}
    for c in sentiments:
        x = [t[0][0] for t in sentiments[c]]
        plot_url[c] = py.iplot(define_hist(x, c)).resource
    
    special = {'Food', 'Service', 'Ambience'}
    top = [cat for cat in sentiments.keys() if cat not in special]
    cloud_food = base_fig + rid + '_food.png'
    cloud_service = base_fig + rid +'_service.png'
    cloud_ambience = base_fig + rid +'_ambience.png'
    clouds = [base_fig + rid + '_' + cat + '.png' for cat in top]

    return render_template('result3a.html', rest_name=rest_name,
                           plot_url=plot_url,
                           rest_names=rest_names, restos=restos,
                           rest_link=rest_link, top=top,
                           rating=rating, sentiments=sentiments,
                           rating_food=rating_food,
                           rating_service=rating_service,
                           rating_ambience=rating_ambience)
    

def define_hist(x, cat):
    xbins_start = -1.0
    xbins_middle = 0.
    xbins_end = 1.0
    xbin_size = 5
    color_pos = 'rgb(106,204,101)'
    color_neg = 'rgb(255,105,97)'

    x_pos = [p for p in x if p >= 0]
    x_neg = [p for p in x if p < 0]

    data = Data([Histogram(x=x_pos, marker=Marker(color=color_pos),
                  name="<b>{}</b>".format(cat),
                  xbins=XBins(start=xbins_middle, end=xbins_end,
                              size=xbin_size)),
                Histogram(x=x_neg, marker=Marker(color=color_neg),
                  name="<b>{}</b>".format(cat),
                  xbins=XBins(start=xbins_start, end=xbins_middle,
                              size=xbin_size)),
            ])

    x_range = [-1, 1]

    axis_style=dict(tickfont=Font(size=14),   # font size (default is 12)
                    titlefont=Font(size=14),  # title font size (default is 12)
                    zeroline=False,           # remove thick zero line
                    autotick=True            
                    )

    layout = Layout(
                    xaxis=XAxis(axis_style,               # style options
                                range=x_range,
                                title='<b>Sentiment [-1 to +1]</b>'
                                ),  
                    yaxis=YAxis(
                        axis_style,               # sytle options
                        title='<b>Snippet count</b>'      # y-axis title 
                        ),
                    showlegend=False,
                    bargap=0.01
    ) 

    fig = Figure(data=data, layout=layout)
    return fig

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