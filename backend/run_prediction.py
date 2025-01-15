import joblib
import pandas as pd

def predict_from_model(model_path, new_data) :
    
    # Load the saved model from file
    model = joblib.load(model_path)

    # check inputs and return errors
    if (new_data['becca_partner'].rsplit('_', 1)[0] == new_data['julianna_partner'].rsplit('_', 1)[0]) and (new_data['becca_partner'] != ''):
        return "Must choose different partners."
    
    # Set variables
    if new_data['becca_partner'] == '' :
        new_data['becca_partner'] = 'None'

    if new_data['julianna_partner'] == '' :
        new_data['julianna_partner'] = 'None'
    
    # Create data for model (start with non-dummy vars)
    X = pd.DataFrame({'game_num': new_data['game_num'], 'becca_start': [True if new_data['serve'] == "Becca" else False]})

    # Create dummies
    for feature in model.feature_names_in_.tolist() :
        if feature not in X.columns.tolist() :
            X[feature] = 0  # all set to 0 first
        else :
            pass

    # Fill in dummies

    # partners
    julianna_partner = 'Player_A_2_' + new_data['julianna_partner'].rsplit('_', 1)[0].capitalize()
    becca_partner = 'Player_B_2_' + new_data['becca_partner'].rsplit('_', 1)[0].capitalize()

    X[julianna_partner] = 1
    X[becca_partner] = 1

    # Predict
    return model.predict(X)[0]