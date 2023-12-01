#Code used from original source at the following link: https://gibberblot.github.io/rl-notes/intro.html

from collections import defaultdict
from qfunction import QFunction


class QTable(QFunction):
    def __init__(self, default=0.0):
        self.qtable = defaultdict(lambda: default)

    def update(self, state, action, delta):
        self.qtable[(state, action)] = round(self.qtable[(state, action)] + delta,4)#.round(4)

    def get_q_value(self, state, action):
        return self.qtable[(state, action)]
