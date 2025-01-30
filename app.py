from flask import Flask, render_template, request, jsonify, json, redirect, url_for
from backend import utils
from backend import run_prediction

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('main.html')

# if a password is needed, replace main.html above with index.html and uncomment below

# @app.route('/authenticate', methods=['POST'])
# def authenticate():
#     # Get the password entered by the user
#     password = request.form['password']
    
#     # Hardcoded password (replace this with your actual password)
#     correct_password = 'jojosiwa'
    
#     # Validate the password
#     if password == correct_password:
#         # Redirect to the protected page if the password is correct
#         return redirect(url_for('pickleball_predictor'))
#     else:
#         # Redirect back to the login page with a message if the password is incorrect
#         return redirect(url_for('index'))

# # main page
# @app.route('/pickleball-predictor')
# def pickleball_predictor():
#     return render_template('main.html')

# predict winner
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    message = run_prediction.predict_from_model('backend/models/rf.joblib', data)

    return jsonify({'message': message})

if __name__ == '__main__':
    app.run(debug=True)