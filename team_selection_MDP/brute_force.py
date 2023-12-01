#Brute force approach - not feasible to run due to very large branching factor.

#discount_factor = 0.9
from copy import deepcopy

def tree_search(mdp, state, depth=10):
    curr_state = deepcopy(state)
    if depth == 0 or mdp.is_terminal(curr_state):
        return 0, None
    
    actions = mdp.get_actions(curr_state)
    best_reward = float('-inf')
    best_action = None
    
    #print("Depth: ", depth)
    #print("State: ", state)
    #print("Actions: ", actions)
    
    for action in actions:
        expected_reward = 0
        #print("Current action: ", action)
        
        for next_state, probability in mdp.get_successors(curr_state, action):
            reward = mdp.get_reward(curr_state, action, next_state)
            next_expected_reward, _ = tree_search(mdp, next_state, depth-1)
            expected_reward += probability * (reward + next_expected_reward)
            
        if expected_reward > best_reward:
            best_reward = expected_reward
            best_action = action
        #print("Expected reward: ", expected_reward)
            
    return best_reward, best_action