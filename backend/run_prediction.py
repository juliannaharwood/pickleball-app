import joblib
import pandas as pd

def predict_from_model(model_path, new_data) :
    
    # Load the saved model from file
    model = joblib.load(model_path)

    team_1 = new_data['team_1']
    team_2 = new_data['team_2']

    # remove numbers of players
    team_1 = [player.split('_')[0] for player in team_1]
    team_2 = [player.split('_')[0] for player in team_2]
    
    # sort so its alphabetical
    team_1.sort()
    team_2.sort()
    
    # Check that the players in team_1 are distinct from the players in team_2
    if len(set(team_1) & set(team_2)) > 0:
        return "Teams cannot have the same players."
    
    # check that 1 or 2 players are lit up on each team
    elif len(team_1) == 0 or len(team_1) > 2 or len(team_2) == 0 or len(team_2) > 2:
        return "Teams must be one or two players."
    
    # check that becca and julianna are playing
    elif 'julianna' not in team_1 + team_2 or 'becca' not in team_1 + team_2:
        return "Predictor can't decide, it's a toss up!"

    # check if becca and julianna are playing together
    elif team_1 == ['becca','julianna']:
        return "Team 1 wins!"
    
    elif team_2 == ['becca','julianna']:
        return "Team 2 wins!"
    
    # check which team julianna is on
    elif 'julianna' in team_1:
        if len(team_1) == 1:
            julianna_partner = 'Player_A_2_None'
        else:
            julianna_partner = 'Player_A_2_' + [player for player in team_1 if player != 'julianna'][0].capitalize()
        if len(team_2) == 1:
            becca_partner = 'Player_B_2_None'
        else:
            becca_partner = 'Player_B_2_' + [player for player in team_2 if player != 'becca'][0].capitalize()

    else:
        if len(team_2) == 1:
            julianna_partner = 'Player_A_2_None'
        else:
            julianna_partner = 'Player_A_2_' + [player for player in team_2 if player != 'julianna'][0].capitalize()
        if len(team_1) == 1:
            becca_partner = 'Player_B_2_None'
        else:
            becca_partner = 'Player_B_2_' + [player for player in team_1 if player != 'becca'][0].capitalize()

    if 'becca' in team_1:
        if new_data['serve'] == 'team_1':
            becca_start = [True]
        else:
            becca_start = [False]
    else:
        if new_data['serve'] == 'team_2':
            becca_start = [True]
        else:
            becca_start = [False]

    # Create data for model (start with non-dummy vars)
    X = pd.DataFrame({'game_num': new_data['game_num'], 'becca_start': becca_start})

    # Create dummies
    for feature in model.feature_names_in_.tolist() :
        if feature not in X.columns.tolist() :
            X[feature] = 0  # all set to 0 first
        else :
            pass

    # Fill in dummies
    X[julianna_partner] = 1
    X[becca_partner] = 1

    # Predict
    prediction = model.predict(X)[0]
    
    if prediction == True:
        message = 'Team 1 wins!' if 'becca' in team_1 else 'Team 2 wins!'
    else:
        message = 'Team 1 wins!' if 'becca' not in team_1 else 'Team 2 wins!'
    
    return message