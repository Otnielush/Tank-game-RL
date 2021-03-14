from .game_object import TankGame


def step(self):

    environment, reward, done, info = 0,0,0,0

    print('Hi')
    return environment, reward, done, info

setattr(TankGame, 'step', step)