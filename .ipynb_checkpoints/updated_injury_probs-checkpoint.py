import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import math
from datetime import timedelta
import time
import timeit

"""
GET PLAYER DATAFRAMES
"""

team_id=23
freq_mp_injury_players=[]
do_mp_injury_players=[]
do_ms_injury_players=[]
age_players=[]
for i in range(1,20):
    player_df = pd.read_csv('team_selection_MDP/Feature_DF/'+str(team_id)+'/'+str(i)+'.csv')
    freq_mp_injury_players.append(player_df.at[0,'frequency_most_prominent_injury'])
    do_mp_injury_players.append(player_df.at[0,'days_out_most_prominent_injury'])
    do_ms_injury_players.append(player_df.at[0,'days_out_most_serious_injury'])
    age_players.append(player_df.at[0,'age'])

player_df_dict = {1:18,22:18,23:1,24:16,25:2,28:18,29:17,31:16,32:11,33:18,34:18,35:18,36:16,37:15,38:14,39:16,40:16,46:18,55:16,58:3}
player_df = pd.read_csv('team_selection_MDP/Feature_DF/'+str(team_id)+'/'+str(player_df_dict[team_id])+'.csv')
general_features = player_df[['team_id','opp_num_tackles','opp_num_fouls','temp','precipMM','distance','age','opp_team_id']].values
game_date_players = pd.to_datetime(player_df['game_date']).dt.date.values

features = ['team_id', 'dist_covered', 'acute_workload',
       'chronic_workload', 'num_injuries', 'total_days_out', 'num_dribbles',
       'days_out_last_injury', 'frequency_most_prominent_injury',
       'days_out_most_prominent_injury', 'days_out_most_serious_injury',
       'days_since_last_injury', 'injuries_past_twelve_months',
       'opp_num_tackles', 'opp_num_fouls', 'temp', 'precipMM', 'distance',
       'age', 'opp_team_id']

def get_opp_mean_feature_df(player_df):
    opp_mean_feature_df = pd.DataFrame()
    opp_mean_feature_df['opp_team_id'] = player_df['opp_team_id'].unique()
    opp_mean_tackles = []
    opp_mean_fouls = []
    for i in player_df['opp_team_id'].unique():
        opp_tackles = player_df[player_df['opp_team_id']==i]['opp_num_tackles'].mean()
        opp_fouls = player_df[player_df['opp_team_id']==i]['opp_num_fouls'].mean()
        opp_mean_tackles.append(opp_tackles)
        opp_mean_fouls.append(opp_fouls)
    opp_mean_feature_df['opp_num_tackles'] = opp_mean_tackles
    opp_mean_feature_df['opp_num_fouls'] = opp_mean_fouls
    return opp_mean_feature_df

"""
Need to consider players who play for multiple clubs still in fixture schedule.
"""
def prepare_static_features(player_feature_df,games_df, opp_mean_feature_df):
    
    ###Get team fixtures
    player_id = player_feature_df['player_id'].values[0]
    player_team = player_feature_df['team_id'].unique()[0]
    fixtures = pd.read_csv('data/team_data/'+str(player_team)+'_fixtures.csv')[['game_date','Competition','season']]
    #-1fixtures = fixtures[fixtures['season']=='18/19'].drop(['season'],axis=1).reset_index(drop=True)
    fixtures['game_date'] = pd.to_datetime(fixtures['game_date']).dt.date
    player_feature_df['game_date'] = pd.to_datetime(player_feature_df['date']).dt.date
    merged_team_df = pd.merge(fixtures,player_feature_df,on='game_date',how='left').drop('date',axis=1)
    in_game_features = player_feature_df[['dist_covered','num_dribbles']]
    
    ###Get in-game features per min played
    mins_played = player_feature_df['rolling_mins_played_exp']
    in_game_features = in_game_features.div(mins_played, axis=0).mean()
    
    ###Impute in features
    merged_team_df['team_id'] = merged_team_df['team_id'].ffill().bfill()
    merged_team_df['player_id'] = merged_team_df['player_id'].ffill().bfill()
    merged_team_df['age'] = merged_team_df['age'].ffill().bfill()
    merged_team_df['precipMM'] = merged_team_df['precipMM'].ffill().bfill()
    merged_team_df['distance'] = merged_team_df['distance'].ffill().bfill()
    merged_team_df['temp'] = merged_team_df['temp'].ffill().bfill()
    #merged_team_df[['precipMM','distance','temp']] =  merged_team_df[['precipMM','distance','temp']].fillna(merged_team_df.mean())
    
    try:
        ##Impute opponent features
        for i,row in merged_team_df.iterrows():
            if math.isnan(row['opp_team_id']):
                game = games_df[(games_df['date'] == row['game_date']) & ((games_df['home_team_id'] == row['team_id']) | (games_df['home_team_id'] == row['team_id']))]
                if len(game) > 0:
                    teams = game[['home_team_id','away_team_id']].values.flatten()
                    opp_team_id = teams[teams != row['team_id']][0]

                else:
                    opp_team_id = 26
                merged_team_df.loc[i, 'opp_team_id'] = opp_team_id
                merged_team_df.loc[i, 'opp_num_tackles'] = opp_mean_feature_df[opp_mean_feature_df['opp_team_id'] == opp_team_id]['opp_num_tackles'].values[0]
                merged_team_df.loc[i, 'opp_num_fouls'] = opp_mean_feature_df[opp_mean_feature_df['opp_team_id'] == opp_team_id]['opp_num_fouls'].values[0]
    except:
        print("Opponent not worked properly")
            
    ##Impute injury features
    merged_team_df['num_injuries'] = merged_team_df['num_injuries'].dropna().iloc[-1]
    merged_team_df['total_days_out'] = merged_team_df['total_days_out'].dropna().iloc[-1]
    merged_team_df['days_out_last_injury'] = merged_team_df['days_out_last_injury'].dropna().iloc[-1]
    merged_team_df['frequency_most_prominent_injury'] = merged_team_df['frequency_most_prominent_injury'].dropna().iloc[-1]
    merged_team_df['days_out_most_prominent_injury'] = merged_team_df['days_out_most_prominent_injury'].dropna().iloc[-1]
    merged_team_df['days_out_most_serious_injury'] = merged_team_df['days_out_most_serious_injury'].dropna().iloc[-1]
    merged_team_df['injuries_past_twelve_months'] = merged_team_df['injuries_past_twelve_months'].dropna().iloc[-1]
    days_passed = (merged_team_df['game_date'] - merged_team_df['game_date'].min()).dt.days
    try:
        merged_team_df['days_since_last_injury'] = days_passed + merged_team_df['days_since_last_injury'].dropna().iloc[-1]
    except:
        merged_team_df['days_since_last_injury'] = days_passed + 200
    return merged_team_df,in_game_features

def add_ingame_features_using_minutes_rbr(pid,row,rolling_mins_played,dist_ratio,drib_ratio, minutes_dict,inj_dict):
    game_date = game_date_players[row]
    yesterday_date = game_date - timedelta(days=1)
    last_week_date = game_date - timedelta(days=8)
    last_month_date = game_date - timedelta(days=32)
    acute_total = 0
    chronic_total = 0
    for key in minutes_dict:
        if last_week_date <= key <= yesterday_date:
            acute_total+=minutes_dict[key]
            
    for key in minutes_dict:
        if last_month_date <= key <= yesterday_date:
            chronic_total+=minutes_dict[key]
    AW=acute_total * dist_ratio
    CW=chronic_total/4 * dist_ratio
    
    out_features = np.array([[general_features[row,0],rolling_mins_played *dist_ratio,AW,CW,inj_dict['num_injuries'],inj_dict['total_days_out'],inj_dict['days_since_last_injury'],inj_dict['days_out_last_injury'],freq_mp_injury_players[pid-1],do_mp_injury_players[pid-1],do_ms_injury_players[pid-1],inj_dict['days_since_last_injury'],inj_dict['injuries_past_twelve_months'],general_features[row,1],general_features[row,2],general_features[row,3],general_features[row,4],general_features[row,5],age_players[pid-1],general_features[row,6]]])
    return out_features 

def prepare_real_features(player_feature_df,games_df, opp_mean_feature_df):
    
    ###Get team fixtures
    player_id = player_feature_df['player_id'].values[0]
    player_team = player_feature_df['team_id'].unique()[0]
    fixtures = pd.read_csv('data/team_data/'+str(player_team)+'_fixtures.csv')[['game_date','Competition','season']]
    fixtures = fixtures[fixtures['season']=='18/19'].drop(['season'],axis=1).reset_index(drop=True)
    fixtures['game_date'] = pd.to_datetime(fixtures['game_date']).dt.date
    player_feature_df['game_date'] = pd.to_datetime(player_feature_df['date']).dt.date
    merged_team_df = pd.merge(fixtures,player_feature_df,on='game_date',how='left').drop('date',axis=1)
    in_game_features = player_feature_df[['dist_covered','num_dribbles']]
    
    ###Get in-game features per min played
    mins_played = player_feature_df['rolling_mins_played_exp']
    in_game_features = in_game_features.div(mins_played, axis=0).mean()
    
    ###Impute in features
    merged_team_df['team_id'] = merged_team_df['team_id'].ffill().bfill()
    merged_team_df['player_id'] = merged_team_df['player_id'].ffill().bfill()
    merged_team_df['age'] = merged_team_df['age'].ffill().bfill()
    merged_team_df['precipMM'] = merged_team_df['precipMM'].ffill().bfill()
    merged_team_df['distance'] = merged_team_df['distance'].ffill().bfill()
    merged_team_df['temp'] = merged_team_df['temp'].ffill().bfill()
    #merged_team_df[['precipMM','distance','temp']] =  merged_team_df[['precipMM','distance','temp']].fillna(merged_team_df.mean())
    
    try:
        ##Impute opponent features
        for i,row in merged_team_df.iterrows():
            if math.isnan(row['opp_team_id']):
                game = games_df[(games_df['date'] == row['game_date']) & ((games_df['home_team_id'] == row['team_id']) | (games_df['home_team_id'] == row['team_id']))]
                if len(game) > 0:
                    teams = game[['home_team_id','away_team_id']].values.flatten()
                    opp_team_id = teams[teams != row['team_id']][0]

                else:
                    opp_team_id = 26
                merged_team_df.loc[i, 'opp_team_id'] = opp_team_id
                merged_team_df.loc[i, 'opp_num_tackles'] = opp_mean_feature_df[opp_mean_feature_df['opp_team_id'] == opp_team_id]['opp_num_tackles'].values[0]
                merged_team_df.loc[i, 'opp_num_fouls'] = opp_mean_feature_df[opp_mean_feature_df['opp_team_id'] == opp_team_id]['opp_num_fouls'].values[0]
    except:
        print("Opponent not worked properly")
    return merged_team_df,in_game_features