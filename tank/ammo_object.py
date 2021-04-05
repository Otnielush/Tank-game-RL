import numpy as np
from copy import copy
from options.video import FRAME_RATE


ammo_types = ['normal', 'rocket', 'bomb', 'jet', 'laser', 'electricity', 'mine', 'none']
# max_distance - deal 0 dmg, at start - 100%; destroy - can destroy obstacles; artillery_fly - fly over obstacles to target
# explossion - makes area dmg; expl_range - explossion range
ammo_features = ['speed', 'max_distance', 'destroy', 'artillery_fly', 'explossion', 'expl_range']  # distance - at start-max dmg, at end-0
a_normal      = [   4,          5,          True,       False,             False,           0]


class Ammo():
    def __init__(self, tank, id):
        self.PIX_CELL = tank.PIX_CELL
        self.parent = tank
        self.angle = tank.direction_tank+tank.direction_tower
        if self.angle > 1:
            self.angle -= 1
        elif self.angle < -1:
            self.angle += 1

        self.done = False

        for (key, value) in zip(ammo_features, a_normal):
            self.__dict__[key] = value
        self.speed /= FRAME_RATE


        if self.angle > 1 or self.angle < -1:  # 0 - down, 90 - left, 180 - up, 270 - right. Or with minus
            self.angle /= 360
        self.speed_x = np.sin(self.angle*np.pi*2) * self.speed
        self.speed_y = np.cos(-self.angle*np.pi*2) * self.speed
        self.X = tank.X + tank.width/2 + self.speed_x * 3
        self.Y = tank.Y + tank.height/2 + self.speed_y * 3
        self.coords_xy = np.rint(np.array([self.X, self.Y])*self.PIX_CELL).astype(int)

        self.id_game = id
        self.tank_id = tank.id_game
        self.damage = 1.0  # 100%
        self.distance = 0
        self.damaged_target_id = 0
        self.damaged_target_yx = np.array([0,0])
        self.damage_dealed_potencial = 0
        self.target_XY = np.array([0, 0])  # for artillery_fly = True, with multy Pixels

        self.height = 0
        self.width = 0


    def __str__(self):
        atts = self.__dict__
        return 'Ammo\n'+'\n'.join([str(x) + ': ' + str(self.__dict__[x]) for x in atts])


    def move(self, coll_map):
        # remove old coords from collision map
        coll_map[round(self.X*self.PIX_CELL), round(self.Y*self.PIX_CELL), 2] = 0
        old_xy = copy(self.coords_xy)
        self.X += self.speed_x
        self.Y += self.speed_y
        self.coords_xy = np.rint(np.array([self.X, self.Y]) * self.PIX_CELL).astype(int)
        self.distance += self.speed
        hit = False

        # hit obstacles
        tar_id = coll_map[self.coords_xy[0], self.coords_xy[1], 0]
        if tar_id.any() > 0:
            hit = True
            self.damaged_target_id = tar_id.max()
            self.hit()

        # hit tanks
        tar_id = coll_map[self.coords_xy[0], self.coords_xy[1], 1]
        if tar_id.any() > 0:
            hit = True
            # TODO not sure about id
            self.damaged_target_id = tar_id.max()
            self.hit()


        if self.artillery_fly:
            if (self.target_XY - self.coords_xy).sum() < 1:
                hit = True
                self.hit()

        if self.distance >= self.max_distance:
            self.done = True
            self.speed_x = 0
            self.speed_y = 0
            self.speed = 0

        return np.rint(old_xy / self.PIX_CELL).astype(int), old_xy, hit


    def hit(self):
        # TODO add explosion
        self.done = True
        self.damaged_target_xy = self.coords_xy
        self.speed_x = 0
        self.speed_y = 0
        self.speed = 0
        self.damage_dealed_potencial = max(((self.max_distance-self.distance)/self.max_distance) * self.parent.dmg, 0)


