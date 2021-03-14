import numpy as np
from collections import deque
import random

# game object
class TankGame():

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.map_generate()


        # SCORES
        self.score_win          = 5
        self.score_hit          = 1
        self.score_kill         = 2
        self.score_death        = -2
        self.score_kill_assist  = 1
        self.score_exploring    = 1




    def map_generate(self):
        # each layer of map mean:
        # 0 - obstacles (0 - road, 0.2 - bushes, 0.4 - desert, 0.6 - forest, 0.8 - swamp, 1 - rock ) + wall
        # 1 - red team (from 0.1 - 1 type of tanks: simple, freezer, artillery, laser, miner, repairer, heavy)
        # 2 - blue team with same types
        # LAST -  fog of war (not sending)
        self.map = np.zeros((self.width, self.height, 4))


    def reset(self, width=0, height=0, team_size=1):
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



