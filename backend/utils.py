import pandas as pd
import numpy as np

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn import preprocessing

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image

def load_data(path):
    # Load data from CSV file into a DataFrame
    data = pd.read_csv(path, header = None)
    return(data)

def clean_data(data):
    # clean header
    header = data.iloc[1]
    header.iloc[-2:] = ['Paddle2','Incomplete'] # set last two column names
    clean_data = data.iloc[2:]
    clean_data.columns = header

    # rename cols
    clean_data = clean_data.rename(columns = {'Court Time': 'court_time',
                                        'Game End Times': 'game_end_times',
                                        'Game #':'game_num'})

    # Fill nas for empty partners and incomplete
    clean_data.Player_A_2 = clean_data.Player_A_2.fillna("None")
    clean_data.Player_B_2 = clean_data.Player_B_2.fillna("None")
    clean_data.Incomplete = clean_data.Incomplete.fillna("Complete")

    # Drop any rows with missing data
    clean_data = clean_data.dropna()

    # Set datatypes
    clean_data = clean_data.astype({'Day':str, 'Park':str, 'Court':str, 'game_num':str, 'Player_A_1':str,
                            'Player_A_2':str, 'Player_B_1':str, 'Player_B_2':str, 'Start_1':str,
                            'Start_2':str, 'Score_1':int, 'Score_2':int, 'Paddle':str, 'Paddle2':str,
                            'Incomplete':str})                                   

    # Define datetime columns
    clean_data.court_time = pd.to_datetime(clean_data.court_time)
    clean_data.game_end_times = pd.to_datetime(clean_data.game_end_times)
    clean_data.Date = pd.to_datetime(clean_data.Date)

    # Remove white space from strings
    clean_data.Player_A_1 = clean_data.Player_A_1.apply(lambda x: x.strip())
    clean_data.Player_A_2 = clean_data.Player_A_2.apply(lambda x: x.strip())
    clean_data.Player_B_1 = clean_data.Player_B_1.apply(lambda x: x.strip())
    clean_data.Player_B_2 = clean_data.Player_B_2.apply(lambda x: x.strip())
    clean_data.Day = clean_data.Day.apply(lambda x: x.strip())
    clean_data.Park = clean_data.Park.apply(lambda x: x.strip())
    clean_data.Paddle = clean_data.Paddle.apply(lambda x: x.strip())
    clean_data.Paddle2 = clean_data.Paddle2.apply(lambda x: x.strip())

    # Add missing game numbers
    clean_data.game_num = clean_data.groupby(['Date']).cumcount()+1

    # Remove incomplete and games that B/J weren't on opposing teams
    clean_data = clean_data[(clean_data['Player_A_1'] == 'Julianna') & (clean_data['Player_B_1'] == 'Becca')]
    # & (clean_data['Incomplete'] == 'nan') -- for now, leave in incomplete games

    return(clean_data)

def feature_engineer(data):
    # Create game length variable
    data['game_length'] = data.groupby(['Date']).game_end_times.diff().fillna(data.game_end_times - data.court_time)
    
    # Create serve variable
    data['becca_start'] = np.where(data['Start_2']== 'Y', True, False)

    # Create year/month variables
    data['year'] = data.Date.dt.year

    # Create winner variable (outcome)
    data['becca_win'] = data.Score_1 < data.Score_2

    # drop unneeded vars
    data = data.drop(['Date','game_end_times','Start_1','Start_2','Score_1','Score_2','Incomplete'], axis = 1)

    return(data)

def create_test_train(data):
    X = data.drop('becca_win', axis=1)
    y = data['becca_win']

    # create one hot encodings for string vars
    oh = preprocessing.OneHotEncoder()
    X_oh = oh.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_oh, y, test_size=0.2)

    return X_train, X_test, y_train, y_test

def get_data(pickle_clean, game_length):
    sorted = pickle_clean.sort_values(by='Game Length')
    game_lengths = sorted['Game Length']
    if game_length == "short":
        return("Shortest game was", str(game_lengths.iloc[0]))
    elif game_length == "medium":
        return("Average game length is", str(np.mean(game_lengths)))
    elif game_length == "long":
        return("Longest game was", str(game_lengths.iloc[-1]))


