from flask import Flask, request, render_template
# from pymongo import MongoDB


PORT = 8888

app = Flask(__name__)

# client = MongoDB()
# coll = client.opentable.clean

# OUR HOME PAGE
#============================================
@app.route('/')
def home():
    return render_template('home.html')
    # return '<h1> Welcome to this page </h1>'


# QUERY PAGE
#============================================
@app.route('/submit_page')
def submission_page():
    return render_template('submit.html')
    # return '''
    # <form action="/topic" method='POST' >
    #     <input type="text" name="user_input" />
    #     <input type="submit" />
    # </form>
    # '''

# RESULT PAGE
#============================================
@app.route('/topic', methods=['POST', 'GET'])
def predict_page():
    # get data from request form, the key is the name you set in your form
    # TODO: 'Sorry, the Critic has not yet visited this restaurant'
    rest_name = request.form['user_input']
    # rest_link = coll.find({'rest_name': rest_name}, {'url': 1, '_id': 0})
    rest_link = 'http://www.opentable.com/lardoise'
    return render_template('result2.html',
                           rest_name=rest_name,
                           rest_link=rest_link)

@app.route('/food', methods=['POST'])
def detail_food():
    rest_name = request.form['user_input']
    # rest_link = coll.find({'rest_name': rest_name}, {'url': 1, '_id': 0})
    rest_link = 'http://www.opentable.com/lardoise'
    return render_template('result_food.html',
                           rest_name=rest_name,
                           rest_link=rest_link)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)