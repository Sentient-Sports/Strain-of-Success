### Match Prediction Model for Section 5

import pickle
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler

#Load the Model
file_path = 'data/predictions/PoissonModel.pickle'
with open(file_path, 'rb') as file:
    # Load the object from the file
    loaded_model = pickle.load(file)
    
#Get Predictions
def get_predictions(X_test):
    X_test,_,_,_ = prep_maher_data(X_test)
    preds=[]
    y_test = predict_maher(loaded_model, X_test)
    pred_df = pd.concat([X_test, y_test], axis='columns')
    preds.append(pred_df)
    preds_df = pd.concat(preds, axis='rows')
    scorelines = calc_maher_scoreline_probs(preds_df)
    #match_probs = get_match_probs(scorelines)[['game_id','team_id_a','team_id_vs_a','team_vaep_a','team_vaep_vs_a','HW','DR','AW']]
    #match_probs['xP_home'] = 3*match_probs['HW']+match_probs['DR']
    #match_probs['xP_away'] = 3*match_probs['AW']+match_probs['DR']
    return scorelines#match_probs

"""
===========================================================================================================
"""
#Maher Model

def predict_maher(model, df):
    y_pred = pd.Series(model.predict(df), name='rate', index=df.index)
    return y_pred

def prep_maher_data(df, date=None):
    df_swapped = df.rename(columns={'team_id':'team_id_vs','team_id_vs':'team_id','away_score':'score','team_vaep':'team_vaep_vs','team_vaep_vs':'team_vaep'})
    df_swapped['is_home']=~df_swapped['is_home']
    #df_swapped['fatigue_diff']=-df['fatigue_diff']
    df=df.rename(columns={'home_score':'score'})
    df=pd.concat([df,df_swapped],ignore_index=True).drop(['home_score','away_score'],axis=1)
    X = df.loc[:, ['game_id','team_id', 'team_id_vs', 'is_home','team_vaep','team_vaep_vs']]
    y=df['score']
    w = None
    team_weights = None

    X['venue'] = 'away'
    X['venue'].mask(X['is_home'], 'home', inplace=True)
    X = X.drop(columns=['is_home'])

    if date is not None:
        w = np.exp((date - df['kickoff']).dt.days * -0.0023).rename('weight')
        team_weights = pd.concat([X.loc[:, ['team_id']].rename(columns={'team_id_a': 'team_id'}), w],
                                 axis='columns').groupby('team_id').sum()
        w /= w.sum()

    return X, y, w, team_weights

# Construct match-focused data frame by joining both sides into the same row
def calc_maher_scoreline_probs(preds_df):
    mask = preds_df['venue'] == 'home'
    scoreline_probs_df = preds_df[mask].merge(
        right=preds_df[~mask],
        on=['game_id'],
        suffixes=['_a', '_b'],
        validate='one_to_one'
    ).merge(
        right=pd.DataFrame({
            'goals_a': range(0, 15)
        }),
        how='cross'
    ).merge(
        right=pd.DataFrame({
            'goals_b': range(0, 15)
        }),
        how='cross'
    )
    scoreline_probs_df['p'] = poisson.pmf(scoreline_probs_df['goals_a'], scoreline_probs_df['rate_a'])
    scoreline_probs_df['p'] *= poisson.pmf(scoreline_probs_df['goals_b'], scoreline_probs_df['rate_b'])
    scoreline_probs_df = scoreline_probs_df[scoreline_probs_df["p"] > 1e-5]
    return scoreline_probs_df

def get_match_probs(scorelines):   
    final_preds_df = scorelines.drop_duplicates('game_id').reset_index(drop=True).copy()
    scorelines.groupby
    match_preds = []
    for game_id in scorelines['game_id'].unique():
        HW=0
        DR=0
        AW=0
        for i,row in scorelines[scorelines['game_id']==game_id].iterrows():
            if row['goals_a']>row['goals_b']:
                HW+=row['p']
            elif row['goals_a']==row['goals_b']:
                DR+=row['p']
            else:
                AW+=row['p']
        match_preds.append([HW,DR,AW])
    final_preds_df[['HW','DR','AW']] = match_preds
    return final_preds_df
