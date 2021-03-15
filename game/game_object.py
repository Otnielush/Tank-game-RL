import numpy as np
from collections import deque
import random

# game object
class TankGame():

    def __init__(self, width, height, team_size):
        self.PIXELS_IN_CELL = 5
        self.team_size = team_size
        self.width = width
        self.height = height
        self.map_obs = {'land': 0, 'bush': 0.14, 'desert': 0.29, 'forest': 0.43, 'water': 0.57, 'swamp': 0.71, 'wall': 0.86, 'rock': 1}   # Dictionary for obstacles
        self.tank_type = {'none': 0, 'simple':0.14, 'freezer':0.29, 'artillery':0.43, 'laser':0.57, 'miner':0.71, 'repairer':0.86, 'heavy':1}

        self.map_generate()



        # SCORES
        self.score_win          = 5
        self.score_hit          = 1
        self.score_kill         = 2
        self.score_death        = -2
        self.score_kill_assist  = 1
        self.score_exploring    = 1
        self.friendly_fire      = -1





    def map_generate(self):
        # each layer of map mean:
        # 0 - obstacles (0 - land, 0.2 - bushes, 0.4 - desert, 0.6 - forest, 0.8 - swamp, 1 - rock ) + wall
        # 1 - red team (from 0.1 - 1 type of tanks: simple, freezer, artillery, laser, miner, repairer, heavy)
        # 2 - blue team with same types
        # LAST -  fog of war (not sending)
        self.map = np.zeros((self.width*self.PIXELS_IN_CELL, self.height*self.PIXELS_IN_CELL, 4))




    def reset(self, width=0, height=0, team_size=0):
        if team_size != 0:
            self.team_size = team_size
        if width == 0 or height == 0:
            width = self.width
            height = self.height

        self.map_generate()

        return self.map




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



