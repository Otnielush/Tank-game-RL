import numpy as np


ammo_types = ['normal', 'rocket', 'bomb', 'jet', 'laser', 'electricity', 'mine', 'none']
ammo_features = ['speed', 'max_distance', 'destroy', 'artillery_fly', 'explossion', 'expl_range']  # distance - at start-max dmg, at end-0
a_normal      = [   3,          8,          True,       False,             False,           0]


class Ammo():
    def __init__(self, tank, id, y, x, angle):
        self.parent = tank
        self.angle = angle

        for (key, value) in zip(ammo_features, a_normal):
            self.__dict__[key] = value

        if angle > 1 or angle < -1:  # 0 - down, 90 - left, 180 - up, 270 - right. Or with minus
            angle /= 360
        self.speed_x = np.sin(angle*np.pi*2) * self.speed
        self.speed_y = np.cos(angle*np.pi*2) * self.speed
        self.speed_YX = np.array([self.speed_y, self.speed_x], dtype=np.float)
        self.pos_YX = np.array([y, x], dtype=np.float)

        self.id = id
        self.tank_id = tank.id
        self.damage = 1.0  # 100%
        self.distance = 0



    def __str__(self):
        atts = self.__dict__
        return 'Ammo\n'+'\n'.join([str(x) + ': ' + str(self.__dict__[x]) for x in atts])

    def move(self):
        self.pos_YX += self.speed_YX
        self.distance += self.speed

    def hit(self):
        return abs((self.max_distance-self.distance)/self.max_distance) * self.parent.dmg


