from .game_object import TankGame
import numpy as np

def build_env_map_team(self, team_num):
    print('Song song song', team_num)
    return np.ones((self.height, self.width, 4))

setattr(TankGame, 'build_env_map_team', build_env_map_team)