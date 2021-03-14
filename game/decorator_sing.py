from .game_object import TankGame


def sing(self):
    print('Song song song')

setattr(TankGame, 'sing', sing)