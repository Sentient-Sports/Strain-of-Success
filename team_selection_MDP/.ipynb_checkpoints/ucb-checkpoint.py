#Code used from original source at the following link: https://gibberblot.github.io/rl-notes/intro.html

import math
import random
from multi_armed_bandit import MultiArmedBandit

class UpperConfidenceBounds(MultiArmedBandit):
    def __init__(self):
        # number of times node has been visited
        self.total = 0
        # number of times node/action pair has been visited
        self.times_selected = {}

    def select(self, state, actions, qfunction):

        # First execute each action one time
        for action in actions:
            if (state[:5],action) not in self.times_selected.keys():
                self.times_selected[(state[:5],action)] = 1
                self.total += 1
                return action

        
        max_actions = []
        max_value = float("-inf")
        # Exploration: Number of simulations completed / Number of times an action is taken. UCB1 parameter.
        for action in actions:
            value = qfunction.get_q_value(state, action) + math.sqrt(
                (0.2 * math.log(self.total)) / self.times_selected[(state[:5],action)]
            )
            if value > max_value:
                max_actions = [action]
                max_value = value
            elif value == max_value:
                max_actions += [action]

        # if there are multiple actions with the highest value
        # choose one randomly
        result = random.choice(max_actions)
        self.times_selected[(state[:5],result)] = self.times_selected[(state[:5],result)] + 1
        self.total += 1
        return result