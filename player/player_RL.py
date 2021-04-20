from numpy import argmax
from .player_superclass import player_obj
from collections import deque
# import tensorflow as tf


# Main RL player
# input neural network model
# TODO: take parent from AI and write action function for RL
class player_RL(player_obj):
    def __init__(self, name):
        super(player_RL, self).__init__(name)

    def done(self):
        super(player_RL, self).done()
        print('done')

    # TODO specify for RL
    def __str__(self):
        return self.id

    # input data/info/environment/status of game
    # output action to make move
    def take_action(self, environment):
        return argmax(self.model.predict(environment))



    # TODO func for save env and rewards




# save all actions and data into game object or may be each player will save it himself
class ReplayBuffer():

    def __init__(self, max_size):
        self.max_size = max_size
        self.transitions = deque()

    def add(self, observation, action, reward, observation2):
        if len(self.transitions) > self.max_size:
            self.transitions.popleft()
        self.transitions.append((observation, action, reward, observation2))

    def sample(self, count):
        # samples = deque(list(self.transitions)[len(self.transitions)-count:])
        # self.transitions = deque(list(self.transitions)[:len(self.transitions)-count], self.max_size)
        # return random.sample(samples, count)

        # return random.sample(self.transitions, count)

        # taking a random part of buffer
        buffer_start = random.randint(0, len(self.transitions) - count)
        return deque(itertools.islice(self.transitions, buffer_start, buffer_start + count))

    def size(self):
        return len(self.transitions)




