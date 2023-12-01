#Code adapted from original source at the following link: https://gibberblot.github.io/rl-notes/intro.html
#Section 4 of the Paper - Includes Progressive Widening from Section 4.2.

from mcts import Node
from mcts import MCTS
from copy import deepcopy
import random
import time
import MDP
import math

#Node for a current state - stores the parent, the state, and the reward from this node.
class SingleAgentNode(Node):
    def __init__(
        self,
        mdp,
        parent,
        state,
        qfunction,
        ucb,
        reward=0.0,
        action=None,
    ):
        super().__init__(mdp, parent, state, qfunction, ucb, reward, action)
        """ Action is the action to get to the child - not the action taken by the child"""
        
        # A dictionary from actions to a set of node-probability pairs
        self.children = {}
        #self.state = deepcopy(state)

    """ Return true if and only if all child actions have been expanded """

    def is_fully_expanded(self):
        hashable_state = (self.state[1],frozenset((k, v) for d in [self.state[2],self.state[4]] for k, v in d.items()))
        node_visits = Node.visits[hashable_state]
        max_nodes_widening = int(1+node_visits**0.5)
        valid_actions = self.sorted_actions[:max_nodes_widening]
        if len(valid_actions) == len(self.children):
            return True
        else:
            return False
    
    """ Select a node that is not fully expanded """
    #Selects a node that isnt fully expanded and will then expand it. If not, choose an action to go to another node.
    def select(self,root_node_game):
        curr_game = self.state[1]
        if self.mdp.is_terminal(self.state) or not self.is_fully_expanded() or ((curr_game-root_node_game)>3):
            #Check if current state has been fully expanded. If it has, move onto another state at a depth lower
            return self
        else:
            #get list of actions based on children of the fully expanded node
            actions = list(self.children.keys())
            #Select an action to follow using the UCB heuristic
            hashable_state = (self.state[1],frozenset((k, v) for d in [self.state[2],self.state[4]] for k, v in d.items()))
            action = self.ucb.select(hashable_state,actions,self.qfunction)
            outcome = self.get_outcome_child(action).select(root_node_game)
            return outcome

    """ Expand a node if it is not a terminal node """
    #Randomly choose an action to expand.
    def expand(self):
        if not self.mdp.is_terminal(self.state):
            # Randomly select an unexpanded action to expand
            hashable_state = (self.state[1],frozenset((k, v) for d in [self.state[2],self.state[4]] for k, v in d.items()))
            node_visits = Node.visits[hashable_state]
            max_nodes_widening = int(1+node_visits**0.5)
            valid_actions = self.sorted_actions[:max_nodes_widening]
            actions = list(map(tuple, valid_actions)) - self.children.keys()
            if len(actions) == 0:
                return self
            action = random.choice(list(actions))

            self.children[action] = []
            return self.get_outcome_child(action)
        return self

    """ Backpropogate the reward back to the parent node """

    def back_propagate(self, reward, child):
        action = child.action
        hashable_state = (self.state[1],frozenset((k, v) for d in [self.state[2],self.state[4]] for k, v in d.items()))
        Node.visits[hashable_state] = Node.visits[hashable_state] + 1
        Node.visits[(hashable_state, action)] = Node.visits[(hashable_state, action)] + 1
        q_value = self.qfunction.get_q_value(hashable_state, action)
        
        #The impact of the times it has been visited changes the amount it updates the q-value by. Early failures may lead to lower reward than late failures?
        delta = ((1 / (Node.visits[(hashable_state, action)])) * (
            reward - self.qfunction.get_q_value(hashable_state, action)
        ))
        self.qfunction.update(hashable_state, action, delta)

        if self.parent != None:
            self.parent.back_propagate(self.reward + reward, self)

    """ Simulate the outcome of an action, and return the child node """
    def get_outcome_child(self, action):
        # Choose one outcome based on transition probabilities. Over time, it should get the other outcomes.
        next_state = self.mdp.state_transition(self.state, action)
        hashable_next_state = (next_state[1],frozenset((k, v) for d in [next_state[2],next_state[4]] for k, v in d.items()))
        reward = self.mdp.get_reward(self.state, action, next_state)

        # Find the corresponding state and return if this already exists
        for (child, _) in self.children[action]:
            hashable_child_state = (child.state[1],frozenset((k, v) for d in [child.state[2],child.state[4]] for k, v in d.items()))
            if hashable_next_state == hashable_child_state:
                return child
                
        # This outcome has not occured from this state-action pair previously
        new_child = SingleAgentNode(
            self.mdp, self, next_state, self.qfunction, self.ucb, reward, action #Action taken is the action to get to the child
        )

        """minutes_action = MDP.create_binary_indicator_list(action,self.mdp.n_players)
        total_prob=1
        for i in range(1,self.mdp.n_players+1):
            if self.state[3][i] == True:
                prob=1
            elif next_state[3][i] == False:
                prob=1-((self.state[2][i]*minutes_action[i-1])/70)
            elif next_state[3][i] == True:
                if len(next_state[0]) == 0:
                    prob = (self.state[2][i]*minutes_action[i-1])/70
                else:
                    prob = ((self.state[2][i]*minutes_action[i-1])/70) / len(next_state[0])
            total_prob *= prob"""
            
        self.children[action] += [(new_child, 0)]
        return new_child
    
class SingleAgentMCTS(MCTS):
    #Creates the initial root node if one hasnt been provided. This will be the initial state of the MDP.
    def create_root_node(self):
        return SingleAgentNode(
            self.mdp, None, self.mdp.get_initial_state(), self.qfunction, self.ucb
        )