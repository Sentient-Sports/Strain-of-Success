import updated_injury_probs as uip
from copy import deepcopy
import pandas as pd
import pickle

predictions_df = pd.read_csv('data/predictions/injury_predictions_df_2.csv')
games_df = pd.read_csv('data/overview_data/games_data.csv')
games_df = games_df.rename(columns={'game_date':'date'})
games_df_comparison = games_df.copy()
games_df_comparison['date'] = pd.to_datetime(games_df_comparison['date']).dt.date

def load_player_data(player_name):
    player_df = predictions_df[predictions_df['player_name'] == player_name][['date','team_id','player_name','days_since_last_injury','num_dribbles','rolling_days_diff_exp','acute_workload','dist_covered','chronic_workload','num_injuries','injury_prob','injured']]
    player_games = games_df[((games_df['home_team_id'] == player_df['team_id'].iloc[0]) | (games_df['away_team_id'] == player_df['team_id'].iloc[0]))].merge(player_df,how='right',on='date')
    game_ids = player_games['game_id']
    opp_mean_feature_df = uip.get_opp_mean_feature_df(predictions_df)
    injury_model = pickle.load(open('data/predictions/injury_model_efficient.sav', 'rb'))
    injury_scaler = pickle.load(open('data/predictions/injury_scaler_efficient.sav', 'rb'))
    feature_data = predictions_df[predictions_df['player_name']==player_name][['date','player_id','team_id','dist_covered', 'acute_workload', 'chronic_workload', 'num_injuries', 'total_days_out','num_dribbles',\
           'days_out_last_injury','frequency_most_prominent_injury', 'days_out_most_prominent_injury',\
           'days_out_most_serious_injury','days_since_last_injury',\
           'injuries_past_twelve_months','opp_num_tackles', 'opp_num_fouls','temp', 'precipMM',
           'distance', 'age', 'rolling_mins_played_exp', 'opp_team_id']]
    feature_data[['team_id','opp_team_id']] = feature_data[['team_id','opp_team_id']].astype("category")
    player_df_simulated,in_game_features = uip.prepare_static_features(feature_data,games_df_comparison,opp_mean_feature_df)
    return (player_df_simulated, in_game_features, injury_model, injury_scaler)

def load_player_data_RL(player_name):
    player_df = predictions_df[predictions_df['player_name'] == player_name][['date','team_id','player_name','days_since_last_injury','num_dribbles','rolling_days_diff_exp','acute_workload','dist_covered','chronic_workload','num_injuries','injury_prob','injured']]
    player_games = games_df[((games_df['home_team_id'] == player_df['team_id'].iloc[0]) | (games_df['away_team_id'] == player_df['team_id'].iloc[0]))].merge(player_df,how='right',on='date')
    game_ids = player_games['game_id']
    opp_mean_feature_df = uip.get_opp_mean_feature_df(predictions_df)
    injury_model = pickle.load(open('data/predictions/injury_model_efficient.sav', 'rb'))
    injury_scaler = pickle.load(open('data/predictions/injury_scaler_efficient.sav', 'rb'))
    feature_data = predictions_df[predictions_df['player_name']==player_name][['date','player_id','team_id','dist_covered', 'acute_workload', 'chronic_workload', 'num_injuries', 'total_days_out','num_dribbles',\
           'days_out_last_injury','frequency_most_prominent_injury', 'days_out_most_prominent_injury',\
           'days_out_most_serious_injury','days_since_last_injury',\
           'injuries_past_twelve_months','opp_num_tackles', 'opp_num_fouls','temp', 'precipMM',
           'distance', 'age', 'rolling_mins_played_exp', 'opp_team_id']]
    feature_data[['team_id','opp_team_id']] = feature_data[['team_id','opp_team_id']].astype("category")
    player_df_simulated,in_game_features = uip.prepare_real_features(feature_data,games_df_comparison,opp_mean_feature_df)
    return (player_df_simulated, in_game_features, injury_model, injury_scaler)