import numpy as np
import pandas as pd

def load_data(path):
    """
    Load data
    """
    data = pd.read_csv(path, header = None)
    return(data)

def clean_data(data):
    """
    Do necessary cleaning
    """
    # clean header
    header = data.iloc[1]
    header.iloc[-2:] = ['Paddle2', 'Incomplete'] # set last two column names
    
    clean_data = data.iloc[2:]
    clean_data.columns = header

    # rename cols
    clean_data = clean_data.rename(columns = {'Court Time': 'court_time', 'Game End Times': 'game_end_times', 'Game #':'game_num'})

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
    clean_data.Date = pd.to_datetime(clean_data.Date, format='mixed')

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
    clean_data.game_num = clean_data.groupby(['Date']).cumcount() + 1

    # Remove incomplete and games that B/J weren't on opposing teams
    clean_data = clean_data[(clean_data['Player_A_1'] == 'Julianna') & (clean_data['Player_B_1'] == 'Becca')]
    # & (clean_data['Incomplete'] == 'nan') -- for now, leave in incomplete games

    return(clean_data)

def feature_engineer(data):
    """
    Create features for model
    """
    # Create game length variables
    data['game_length'] = data.groupby(['Date']).game_end_times.diff().fillna(data.game_end_times - data.court_time)  # first game will be end time - court start time
    data['game_length_mins'] = data.game_length.dt.total_seconds()/60
    
    # Create serve variable
    data['becca_start'] = np.where(data['Start_2']== 'Y', True, False)

    # Create year, month, hour variables
    data['year'] = data.Date.dt.year
    data['month'] = data.Date.dt.month
    data['hour'] = data.court_time.dt.hour

    # Create winner variable (outcome)
    data['becca_win'] = data.Score_1 < data.Score_2

    # drop unneeded vars
    data = data.drop(['Date', 'game_end_times', 'Start_1', 'Start_2', 'Score_1', 'Score_2', 'Incomplete'], axis = 1)

    return(data)


