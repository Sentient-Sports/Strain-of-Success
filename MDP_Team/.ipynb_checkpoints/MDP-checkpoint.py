#This code holds the Team Formation MDP discussed in Section 3.2

import pandas as pd
import random
import pickle
import numpy as np
import updated_injury_probs as uip
from copy import deepcopy
import copy
import time
import timeit
import xgboost as xgb
from datetime import timedelta
import datetime
import itertools
import daal4py as d4p
from scipy.stats import gaussian_kde
"""
------------------------------------------
"""
#Setup DAAL model
daal_predict_algo = d4p.gbt_classification_prediction(
    nClasses=2,
    resultsToEvaluate="computeClassProbabilities",
    fptype='float'
)

team_id = 23
team_squad = pd.read_csv('MDP_Team/Team_squads/'+str(team_id)+'.csv')
reward_df = pd.read_csv('MDP_Team/Team_rewards/'+str(team_id)+'.csv')
reward_dict = dict(zip(reward_df.columns[6:].values,range(len(reward_df.columns[6:].values))))
reward_matrix = reward_df.iloc[:,6:].values
team_reward_without_reserves = reward_df[[col for col in reward_df.columns if '20' not in col]].copy()
team_ranked_squads = (team_reward_without_reserves.iloc[:,6:].sum() / team_reward_without_reserves.iloc[:,6:].sum().sum()).sort_values(ascending=False).index.tolist()
team_ranked_squads_index = np.array([[int(x)-1 for x in t[1:-1].split(',')][1:] for t in team_ranked_squads])
n_players = 19

with open('MDP_Team/Player_predictions/reserve_dictionary.pkl', 'rb') as f:
    reserve_dictionary = pickle.load(f)
    
with open('MDP_Team/Player_predictions/reserve_dictionary_numbered.pkl', 'rb') as f:
    reserve_dictionary_numbered = pickle.load(f)

def create_binary_indicator_list(original_list, sublist_size):
    binary_indicator = [0] * sublist_size
    for index in original_list:
        if index >= sublist_size:
            continue
        binary_indicator[index] = 90
    return binary_indicator

injury_data = injury_data = pd.read_csv('injury_data/all_player_injuries_updated.csv')
inj_kde = gaussian_kde(injury_data['Days'])
injury_lengths_array = inj_kde.resample(50000)[0]
injury_lengths_array[injury_lengths_array < 0] = 0
injury_lengths_array = injury_lengths_array.tolist()
"""
-----------------------------------------
"""

class GameMDP:
    def __init__(self, n_games, n_players, injury_model, game_dates, player_features,scaler_mean,scaler_var,inj_kde,inj_dict):
        self.n_games = n_games
        self.n_players = n_players
        self.injury_model = injury_model
        self.scaler_mean = scaler_mean
        self.scaler_var = scaler_var
        self.inj_kde = inj_kde
        self.distance_per_mins = {key: value['dist_covered'] for key, value in player_features.items()}
        self.num_dribbles_per_mins = {key: value['num_dribbles'] for key, value in player_features.items()}
        self.inj_dict = inj_dict
        self.current_reward = 0
        
        #Creates player df for values for each game. Updates feature df for each game, and finally injury_features is features used to compute injury prob
        self.game_dates = game_dates
        self.init_state()
        self.current_state = self.get_state()
        self.initial_state = self.current_state
        self.X = None
        
    def init_state(self):
        self.remaining_games = tuple(range(self.n_games))
        self.injury_prob = {i: 0 for i in range(1, self.n_players+1)}
        self.is_injured = {i: False for i in range(1, self.n_players+1)}
        self.games_until_available = {i: 0 for i in range(1, self.n_players+1)}
        self.rolling_mins_played = {i: 0 for i in range(1, self.n_players+1)}
        self.minutes_dict = {i: {} for i in range(1, self.n_players+1)}
        
        self.current_state = (self.remaining_games, self.remaining_games[0], 
                self.injury_prob, self.is_injured, self.games_until_available, self.rolling_mins_played, self.minutes_dict,self.inj_dict)
        self.injury_prob,  self.rolling_mins_played,self.minutes_dict = self.update_injury_prob({i: 0 for i in range(1, self.n_players+1)},True,self.remaining_games[0],self.remaining_games,self.injury_prob,self.rolling_mins_played,self.minutes_dict,self.inj_dict)
        self.current_state = (self.remaining_games, self.remaining_games[0], 
                self.injury_prob, self.is_injured, self.games_until_available, self.rolling_mins_played, self.minutes_dict,self.inj_dict)
        
    def get_ewm(ewm,x):
        if ewm is None:
            ewm=x
        else:
            ewm=0.8*x+0.2*ewm
        return ewm

    def get_state(self):
        return self.current_state
    
    def get_initial_state(self):
        return self.initial_state
    
    def get_curr_reward(self):
        return self.current_reward
    
    def set_state(self, state):
        self.current_state = state
        
    def get_discount_factor(self):
        return 1
    
    def predict_proba(self,x,m,v,model):
        trans_data = (x - m) / v ** .5
        pred = daal_predict_algo.compute(trans_data, model).probabilities[:,1]
        return pred
    
    def filter_lists_with_value(self,lst, value, n):
        return [sublst for sublst in lst if sublst.count(value) == n]
    
    def get_actions(self, state):
        remaining_games, _, _, is_inj, _, _,_,_ = state
        is_injured=is_inj.copy()
        actions = team_ranked_squads_index.copy()
        
        #Remove any which suggests an injured player is playing over 0 minutes
        #for i in range(1,len(is_injured)+1):
        #    if is_injured[i]==True:
        #        actions = [l for l in actions if (i-1) not in l]
        true_keys = []
        for n in is_injured:
            if is_injured[n] == True:
                true_keys.append(n-1)
        index = np.all(~np.isin(actions,true_keys),axis=1)
        actions = actions[index]
        
        if len(actions) == 0:
            return self.get_reserves_actions(is_injured,remaining_games)

        if len(remaining_games) == 0:
            return None
        else:
            return actions
        
    def get_reward(self, state, action, next_state):
        index_tuple = str(tuple([0]+[x+1 for x in action]))
        reward_index = reward_dict[index_tuple]
        reward = reward_matrix[state[1],reward_index]
        self.current_reward += reward
        return reward
        
    def state_transition(self, state, action):
        action = create_binary_indicator_list(action,n_players)
        #print("Action Minutes: ", action)
        result = self.get_successors(state, action)
        return result
        
    def update_injury_prob(self, action, initializing, current_game,remaining_games,injury_prob,rolling_mins_played,minutes_dict,inj_dict):
        #remaining_games, current_game, injury_prob, is_injured, games_until_available, rolling_mins_played, minutes_dict, _ = state
        injury_prob_all = {}
        injury_features_all = {}
        pred_list = []
        
        if initializing == True:
            for i in range(1,self.n_players+1):
                injury_features_all[i] = uip.add_ingame_features_using_minutes_rbr(i,0,rolling_mins_played[i],self.distance_per_mins[i], self.num_dribbles_per_mins[i],minutes_dict[i],inj_dict[i])
                pred_list+=injury_features_all[i].tolist()
            inj_updated = self.predict_proba(pred_list,self.scaler_mean,self.scaler_var,self.injury_model).round(3)
            injury_prob_all = dict(zip(range(1,self.n_players+1),inj_updated))
            #injury_prob_all[i] = self.predict_proba_single(injury_features_all[i],self.scaler_mean,self.scaler_var,self.injury_model).round(4)
            
            return (injury_prob_all, rolling_mins_played,minutes_dict)#(injury_prob, player_df, injury_features)
        elif len(remaining_games)==1:
            return (injury_prob, rolling_mins_played,minutes_dict)
        else:
            for i in range(1,self.n_players+1):
                minutes=action[i-1]
                if current_game == 0:
                    rolling_mins_played[i] = minutes
                else:
                    rolling_mins_played[i] = 0.8*minutes+0.2*rolling_mins_played[i] #Compute updates for rolling average
                game_date = self.game_dates[current_game]
                minutes_dict[i][game_date] = minutes
                injury_features_all[i] = uip.add_ingame_features_using_minutes_rbr(i,current_game+1,rolling_mins_played[i],self.distance_per_mins[i], self.num_dribbles_per_mins[i], minutes_dict[i],inj_dict[i])
                pred_list+=injury_features_all[i].tolist()
            inj_updated = self.predict_proba(pred_list,self.scaler_mean,self.scaler_var,self.injury_model).round(3)
            injury_prob_all = dict(zip(range(1,self.n_players+1),inj_updated))
            return (injury_prob_all, rolling_mins_played,minutes_dict) #(injury_prob+(action/10000),0,0,0)
        
    
    def predict_injury(self, injury_prob):
        return random.random() < injury_prob
    
    def is_terminal(self, state):
        remaining_games, current_game, injury_prob, is_injured, games_until_available, rolling_mins_played,minutes_dict, inj_dict = state
        return len(remaining_games) == 0
    
    def get_successors(self, state, action):
        remaining_games, current_game, i_prob, is_inj, games_until_available, r_mins_played,m_dict, i_dict = state
        inj_dict = {key: dict(value) for key, value in i_dict.items()}
        minutes_dict = {key: dict(value) for key, value in m_dict.items()}
        rolling_mins_played = r_mins_played.copy()
        injury_prob_prelim = i_prob.copy()
        is_injured = is_inj.copy()
        new_remaining_games = tuple(remaining_games[1:])
        next_game = remaining_games[1] if len(remaining_games) > 1 else None
        
        #Mean minutes played = ~72.5
        danger=3
        if current_game % 7 == 0: # * danger
            injury_prob = {i: injury_prob_prelim[i] * (action[i-1]/70) * 3 for i in range(1, self.n_players+1)}
        else:
            injury_prob = {i: injury_prob_prelim[i] * (action[i-1]/70) * 3 for i in range(1, self.n_players+1)}
        random_nums = [random.random() for _ in range(self.n_players)]
        
        # Handle terminal state
        if self.is_terminal(state):
            return state
        
        new_is_injured_all={}
        new_games_until_available_all={}
        
        for i in range(1,self.n_players+1):
            curr_action = action[i-1]
            # Handle action 0: rest
            if curr_action == 0:
                if is_injured[i]:
                    if ((games_until_available[i]-1) == 0):
                        is_injured[i] = False
                        if next_game != None:
                            inj_dict[i]['days_since_last_injury']+=(self.game_dates[next_game]-self.game_dates[current_game]).days
                else:
                    if next_game != None:
                        inj_dict[i]['days_since_last_injury']+=(self.game_dates[next_game]-self.game_dates[current_game]).days
                new_is_injured_all[i] = is_injured[i]
                new_games_until_available_all[i] = max(games_until_available[i]-1,0)
            # Handle action 1: play
            else:
                if random_nums[i-1] >= injury_prob[i]:
                    new_is_injured_all[i] = False
                    new_games_until_available_all[i] = 0 if next_game is not None else None
                    if next_game != None:
                        inj_dict[i]['days_since_last_injury']+=(self.game_dates[next_game]-self.game_dates[current_game]).days
                else:
                    new_is_injured_all[i] = True 
                    current_game_date = self.game_dates[current_game]
                    days_out = timedelta(days=random.choice(injury_lengths_array))
                    return_date = min(self.game_dates, key=lambda date: abs(date - (current_game_date + days_out)))
                    new_days_out = (return_date-current_game_date).days
                    games_injured = len(self.game_dates[(self.game_dates > current_game_date) & (self.game_dates <= return_date)])
                    if games_injured == 0:
                        new_is_injured_all[i] = False
                        new_games_until_available_all[i] = 0 if next_game is not None else None
                        if next_game != None:
                            inj_dict[i]['days_since_last_injury']+=(self.game_dates[next_game]-self.game_dates[current_game]).days
                    else:
                        if next_game == None:
                            new_games_until_available_all[i] = None
                        elif next_game + games_injured > new_remaining_games[-1]:
                            new_games_until_available_all[i] = len(new_remaining_games)
                        else:
                            new_games_until_available_all[i] = games_injured
                        inj_dict[i]['days_since_last_injury'] = 0
                        inj_dict[i]['days_out_last_injury']=new_days_out
                        inj_dict[i]['num_injuries']+=1
                        inj_dict[i]['total_days_out']+=new_days_out
                        inj_dict[i]['injuries_past_twelve_months']+=1
         
        new_injury_prob, new_rolling_mins_played,new_minutes_dict = self.update_injury_prob(action,False,current_game,remaining_games,injury_prob_prelim,rolling_mins_played,minutes_dict,inj_dict)
        next_state = (new_remaining_games, next_game, new_injury_prob, new_is_injured_all, new_games_until_available_all,new_rolling_mins_played,new_minutes_dict, inj_dict)
        return next_state
            
    def get_reserves_actions(self,is_injured,remaining_games):

        if len(remaining_games) == 0:
            return None
        else:
            num_injured_cbs = sum([is_injured.get(key) for key in [1,2,3,4]])
            num_injured_rbs = sum([is_injured.get(key) for key in [5,6]])
            num_injured_lbs = sum([is_injured.get(key) for key in [7,8]])
            num_injured_cms = sum([is_injured.get(key) for key in [9,10,11,12,13]])
            num_injured_rws = sum([is_injured.get(key) for key in [14,15]])
            num_injured_lws = sum([is_injured.get(key) for key in [16,17]])
            num_injured_sts = sum([is_injured.get(key) for key in [18,19]])
            indexes=[]

            reserved_positions_dict = dict() 
            count=0
            if num_injured_cbs == 4:
                count+=2
                indexes+=[0,1]
            elif num_injured_cbs == 3:
                count+=1
                indexes+=[1]
            if num_injured_rbs == 2:
                count+=1
                indexes+=[2]
            if num_injured_lbs == 2:
                count+=1
                indexes+=[3]
            if num_injured_cms == 5:
                count+=3
                indexes+=[4,5,6]
            elif num_injured_cms == 4:
                count+=2
                indexes+=[5,6]
            elif num_injured_cms == 3:
                count+=1
                indexes+=[6]
            if num_injured_rws == 2:
                count+=1
                indexes+=[7]
            if num_injured_lws == 2:
                count+=1
                indexes+=[8]
            if num_injured_sts == 2:
                count+=1
                indexes+=[9]

            if num_injured_cbs == 4:
                reserved_positions_dict['cb_injury'] = reserve_dictionary_numbered['two_cb_'+str(count)]
            elif num_injured_cbs == 3:
                reserved_positions_dict['cb_injury'] = reserve_dictionary_numbered['one_cb_'+str(count)]
            elif num_injured_rbs == 2:
                reserved_positions_dict['rb_injury'] = reserve_dictionary_numbered['one_rb_'+str(count)]
            elif num_injured_lbs == 2:
                reserved_positions_dict['lb_injury'] = reserve_dictionary_numbered['one_lb_'+str(count)]
            elif num_injured_cms == 5:
                reserved_positions_dict['cm_injury'] = reserve_dictionary_numbered['three_cm_'+str(count)]
            elif num_injured_cms == 4:
                reserved_positions_dict['cm_injury'] = reserve_dictionary_numbered['two_cm_'+str(count)]
            elif num_injured_cms == 3:
                reserved_positions_dict['cm_injury'] = reserve_dictionary_numbered['one_cm_'+str(count)]
            elif num_injured_rws == 2:
                reserved_positions_dict['rw_injury'] = reserve_dictionary_numbered['one_rw_'+str(count)]
            elif num_injured_lws == 2:
                reserved_positions_dict['lw_injury'] = reserve_dictionary_numbered['one_lw_'+str(count)]
            elif num_injured_sts == 2:
                reserved_positions_dict['st_injury'] = reserve_dictionary_numbered['one_st_'+str(count)]

            true_keys = []
            for n in is_injured:
                if is_injured[n] == True:
                    true_keys.append(n-1)

            # Vertically stack the arrays
            stacked_array = next(iter(reserved_positions_dict.values()))#np.vstack(arrays)
            index = np.all((~np.isin(stacked_array,true_keys)),axis=1)
            a = stacked_array[index]
            index2 = np.all((a[:, indexes] == 19),axis=1)
            a = a[index2]

            return a
            
        