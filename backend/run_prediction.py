import joblib
import pandas as pd

def predict_from_model(model_path, new_data) :
    
    # Load the saved model from file
    model = joblib.load(model_path)

    if new_data['becca_partner'] == '' :
        new_data['becca_partner'] = 'None'

    if new_data['julianna_partner'] == '' :
        new_data['julianna_partner'] = 'None'
    
    # Create data for model (start with non-dummy vars)
    X = pd.DataFrame({'game_num': new_data['game_num'], 'becca_start': [True if new_data['serve'] == "becca" else False]})

    # Create dummies
    for feature in model.feature_names_in_.tolist() :
        if feature not in X.columns.tolist() :
            X[feature] = 0
        else :
            pass

    # Fill in dummies
    julianna_partner = 'Player_A_2_' + new_data['julianna_partner'].capitalize()
    becca_partner = 'Player_B_2_' + new_data['becca_partner'].capitalize()

    X[julianna_partner] = 1
    X[becca_partner] = 1

    # Predict
    return model.predict(X)[0]