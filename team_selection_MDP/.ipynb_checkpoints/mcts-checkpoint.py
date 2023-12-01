#Code adapted from original source at the following link: https://gibberblot.github.io/rl-notes/intro.html
#Contains the MCTS code - Section 6

import math
import time
import random
from collections import defaultdict
import numpy as np
import pickle
import pandas as pd
from datetime import timedelta
from scipy.stats import gaussian_kde

"""
-------------------------------------------------------------------------------------------------------------------------
"""
#Reward function stuff
team_id=23
reward_df = pd.read_csv('team_selection_MDP/Team_rewards/'+str(team_id)+'.csv')
team_reward_without_reserves = reward_df[[col for col in reward_df.columns if '20' not in col]].copy()
feature_df = pd.read_csv('team_selection_MDP/Feature_DF/'+str(team_id)+'/18.csv')
game_dates = pd.to_datetime(feature_df['game_date']).dt.date.values
reward_dict = dict(zip(reward_df.columns[6:].values,range(len(reward_df.columns[6:].values))))
reward_matrix = reward_df.iloc[:,6:].values
team_squad = pd.read_csv('team_selection_MDP/Team_squads/'+str(team_id)+'.csv')
squad_vaeps = team_squad['vaep'].values[1:]
squad_vaep = team_squad['vaep'].values
importances = (team_reward_without_reserves.iloc[:,6:].max(axis=1)-team_reward_without_reserves.iloc[:,6:].mean(axis=1)).values
injury_data = pd.read_csv('data/injury_data/all_player_injuries_updated.csv')
inj_kde = gaussian_kde(injury_data['Days'])
injury_lengths_array = inj_kde.resample(50000)[0]
mean_injury_length = np.mean(injury_lengths_array)

importance_missed_injury = []
for i in range(len(game_dates)):
    current_game_date = game_dates[0]
    return_date = min(game_dates, key=lambda date: abs(date - (current_game_date + timedelta(days=mean_injury_length))))
    game_diff = game_dates.tolist().index(return_date)-game_dates.tolist().index(current_game_date)+1
    future_importance = importances[i:i+game_diff].sum()
    importance_missed_injury.append(future_importance)

with open('team_selection_MDP/Player_predictions/player_counts_dict.pkl', 'rb') as f:
    player_id_counts = pickle.load(f)
    
positions = [[1,2,3,4],[5,6],[7,8],[9,10,11,12,13],[14,15],[16,17],[18,19]]
counts = [2,1,1,3,1,1,1]

def create_binary_indicator_ohe(original_list, sublist_size):
    zeros_array = np.zeros(sublist_size)
    zeros_array[original_list] = 1
    return zeros_array

def greedy_selection(selection_list, player_vaeps, num_selections, injury_dict):
    total_val = 0
    
    new_selection_list = []
    for i in range(len(selection_list)):
        if injury_dict[selection_list[i]] == False:
            new_selection_list.append(selection_list[i])
    
    while len(new_selection_list) < num_selections:
        new_selection_list.append(20)
            
    selection_values = player_vaeps[new_selection_list]
    max_indexes = np.argsort(selection_values)[-num_selections:]
    return [new_selection_list[i]-1 for i in sorted(max_indexes)]


#########
#Heuristic for Progressive Widening - Section 4.2 of the paper
#########
def get_sorted_actions(available_actions, injury_probs, current_game, squad_vaeps):
    if available_actions is None:
        return None
    
    danger=3
    if current_game % 5 == 0:
        inj_probs = np.array(list(injury_probs.values()) + [0]) #* (90/70) #* 2 * danger
    else:
         inj_probs = np.array(list(injury_probs.values()) + [0]) #* (90/70) #* 2
    
    # Calculate the heuristic value for each action
    actions_ohe = np.zeros([len(available_actions),20])
    actions_ohe[np.arange(actions_ohe.shape[0])[:, None], available_actions % 20] = 1
    heur_val = np.sum(
        (actions_ohe * importances[current_game] * squad_vaeps)
        + ((1 - actions_ohe) * (inj_probs * importance_missed_injury[current_game] * squad_vaeps)),
        axis=1
    )
    
    # Sort the actions by heuristic value
    sorted_actions = np.argsort(heur_val)[::-1]
    return available_actions[sorted_actions]

"""
-----------------------------------------------------------------------------------------------------------------------------
"""

class Node:

    # Record a unique node id to distinguish duplicated states
    next_node_id = 0

    # Records the number of times states have been visited
    visits = defaultdict(lambda: 0)

    def __init__(self, mdp, parent, state, qfunction,ucb,reward=0.0, action=None):
        self.mdp = mdp
        self.parent = parent
        self.state = state
        self.qfunction = qfunction
        self.id = Node.next_node_id
        Node.next_node_id += 1

        # The immediate reward received for reaching this state, used for backpropagation
        self.reward = reward
        
        self.ucb = ucb
        self.sorted_actions = get_sorted_actions(mdp.get_actions(state),state[2],state[1],squad_vaeps)
        
        # The action that generated this node
        self.action = action

    """ Select a node that is not fully expanded """

    def select(self): abstract


    """ Expand a node if it is not a terminal node """

    def expand(self): abstract


    """ Backpropogate the reward back to the parent node """

    def back_propagate(self, reward, child): abstract


    """ Return the value of this node """ 
    def get_value(self):
        hashable_state = (self.state[1],frozenset((k, v) for d in [self.state[2],self.state[4]] for k, v in d.items()))
        (_, max_q_value) = self.qfunction.get_max_q(
            hashable_state, self.mdp.get_actions(self.state)
        )
        return max_q_value

    """ Get the number of visits to this state """

    def get_visits(self):
        hashable_state = (self.state[1],frozenset((k, v) for d in [self.state[2],self.state[4]] for k, v in d.items()))
        return Node.visits[hashable_state]


class MCTS:
    def __init__(self, mdp,qfunction,ucb,kde,max_importance):
        self.mdp = mdp
        self.qfunction = qfunction
        self.ucb = ucb
        
    """
    Execute the MCTS algorithm from the initial state given, with timeout in seconds
    """

    def mcts(self, timeout=1, root_node=None):
        #If no root node is provided, then it creates one corresponding to the root node of the tree.
        if root_node is None:
            root_node = self.create_root_node()
        root_node_game = root_node.state[1]
        
        start_time = time.time()
        current_time = time.time()
        total_selection_time = 0
        total_expansion_time = 0
        total_simulation_time = 0
        total_backpropogation_time = 0
        count = 0
        selection_count = 0
        #Anytime algorithm - carry on running MCTS until a certain time limit is reached.
        while current_time < start_time + timeout:
            #Run the *select* stage of the mcts algorithm
            st = time.time()
            selected_node = root_node.select(root_node_game)
            total_selection_time += (time.time()-st)
            selection_count+=1
            #print("Select time: ", time.time()-select_time)
            if not self.mdp.is_terminal(selected_node.state):
                count+=1
                #expand_time = time.time()
                st = time.time()
                child = selected_node.expand()
                total_expansion_time += (time.time()-st)
                #print("Expand time: ", time.time()-expand_time)
                #reward_time = time.time()
                st = time.time()
                reward = self.simulate(child)
                #print("State: ", child.state[1])
                #print("Sim reward: ", reward)
                total_simulation_time += (time.time()-st)
                #print("Reward time: ", time.time()-reward_time)
                #backprop_time = time.time()
                st = time.time()
                selected_node.back_propagate(reward, child)
                total_backpropogation_time += (time.time()-st)
                #print("Backprop time: ", time.time()-backprop_time)
            current_time = time.time()

        print(count)
        print("SC", selection_count)
        return root_node,total_selection_time,total_expansion_time,total_simulation_time,total_backpropogation_time

    """ Create a root node representing an initial state """

    def create_root_node(self): abstract


    """ Choose a random action. Heustics can be used here to improve simulations. """

    def choose(self, state):
        return random.choice(self.mdp.get_actions(state))

    """ Simulate until a terminal state """
    
    def simulate(self, node):
        av_reward = 0
        num_loops = 1
        for i in range(1):   
            state = node.state
            curr_game = state[1]
            cumulative_reward = 0#node.reward
            depth = 0
            while (state[1] is not None) and (state[1] <= curr_game+3) and (not self.mdp.is_terminal(state)):
                #Action chose using value heuristics from IRL games
                action=[]
                for p,c in zip(positions,counts):
                    action += greedy_selection(p,squad_vaep,c,state[3])
                # Execute the action
                next_state = self.mdp.state_transition(state, action)
                reward = self.mdp.get_reward(state, action, next_state)
                
                cumulative_reward += reward
                depth += 1

                state = next_state
            if self.mdp.is_terminal(state):
                cumulative_reward += node.reward
                av_reward += cumulative_reward
                continue
                
            new_dict = {k: v for k, v in state[4].items()}
            new_inj_dict = {key: value > 0 for key, value in new_dict.items()}
            changed_dict = True
            for i in range(state[1],state[0][-1]+1):
                if changed_dict:
                    action=[]
                    for p,c in zip(positions,counts):
                        action += greedy_selection(p,squad_vaep,c,new_inj_dict)
                    index_tuple = str(tuple([0]+[x+1 for x in action]))
                    reward_index = reward_dict[index_tuple]
                    
                reward = reward_matrix[i,reward_index]
                new_dict = {k: max(0,v - 1) for k, v in new_dict.items()}
                temp_new_inj_dict = {key: value > 0 for key, value in new_dict.items()}
                if temp_new_inj_dict != new_inj_dict:
                    new_inj_dict = temp_new_inj_dict
                    changed_dict = True
                else:
                    changed_dict = False
                cumulative_reward += reward
            av_reward += cumulative_reward
            
        av_reward = av_reward / num_loops
        return av_reward
    