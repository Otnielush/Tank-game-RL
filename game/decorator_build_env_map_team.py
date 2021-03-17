from .game_object import TankGame


def build_env_map_team(self):
    print('Song song song')
    return [1,1,1]

setattr(TankGame, 'build_env_map_team', build_env_map_team)