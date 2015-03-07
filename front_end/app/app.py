from flask import Flask, request, render_template


PORT = 8888

app = Flask(__name__)

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
@app.route('/topic', methods=['POST'])
def predict_page():
    # get data from request form, the key is the name you set in your form
    # TODO: 'Sorry, the Critic has not yet visited this restaurant'
    rest_name = request.form['user_input']
    rest_link = 'http://www.opentable.com/lardoise'
    return render_template('result2.html',
                           rest_name=rest_name,
                           rest_link=rest_link)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)