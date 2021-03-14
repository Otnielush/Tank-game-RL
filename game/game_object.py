import numpy as np
from collections import deque
import random

# game object
class TankGame():

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.generate_map()


    def generate_map(self):
        self.map = np.zeros((self.width, self.height))


    def reset(self, width=0, height=0, team_size=1):
        if width == 0 or height == 0:
            width = self.width
            height = self.height

        self.generate_map()

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



