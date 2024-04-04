from flask import Flask, render_template, request, jsonify, json, redirect, url_for
from backend import train
import pandas as pd

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    # Get the password entered by the user
    password = request.form['password']
    
    # Hardcoded password (replace this with your actual password)
    correct_password = 'jojosiwa'
    
    # Validate the password
    if password == correct_password:
        # Redirect to the protected page if the password is correct
        return redirect(url_for('pickleball_predictor'))
    else:
        # Redirect back to the login page with a message if the password is incorrect
        return redirect(url_for('index'))

@app.route('/pickleball-predictor')
def pickleball_predictor():
    return render_template('flask_index.html')

@app.route('/pickleball-predictor/show_data', methods=['POST'])
def show_data():
    pickle_raw = train.load_data()
    pickle_clean = train.clean_data(pickle_raw)
    pickle_clean = train.feature_engineer(pickle_clean)
    data = request.get_json()
    selected_value = data['value']
    # Process the selected value as needed
    final_data = train.get_data(pickle_clean, selected_value)
    print(final_data)
    return jsonify(str(final_data))

if __name__ == '__main__':
    app.run(debug=True)