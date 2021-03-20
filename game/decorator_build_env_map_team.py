from .game_object import TankGame
import numpy as np


# TODO Don`t need more. Change to smthg usefull
def some_func(self, team_num):
    return np.ones((self.height, self.width, 4))

setattr(TankGame, 'some_func', some_func)

