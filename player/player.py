from .player_AI_object import player_AI
from .player_human_object import player_human
from .player_RL_object import player_RL
from collections import deque





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












# import os
#
# dell = '/' if __name__ == '__main__' else '\\'
# print('name:', __name__)
#
# path = dell.join(__file__.split(dell)[:-1])+dell
# files = [f for f in os.listdir(path) if f.endswith('.py') and f != __file__.split(dell)[-1]]
#
# print(files)
#
# for f in files:
#     module_obj = __import__(path+f[:-3])
#     globals()[f[:-3]] = module_obj
#
# print(player_human_object)
#
# pp = player_human_object.player_human(3, 'ser')
#
# print(pp)