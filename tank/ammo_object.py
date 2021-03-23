import numpy as np
from copy import copy
from video.graphics import FRAME_RATE


ammo_types = ['normal', 'rocket', 'bomb', 'jet', 'laser', 'electricity', 'mine', 'none']
# max_distance - deal 0 dmg, at start - 100%; destroy - can destroy obstacles; artillery_fly - fly over obstacles to target
# explossion - makes area dmg; expl_range - explossion range
ammo_features = ['speed', 'max_distance', 'destroy', 'artillery_fly', 'explossion', 'expl_range']  # distance - at start-max dmg, at end-0
a_normal      = [   4,          8,          True,       False,             False,           0]


class Ammo():
    def __init__(self, tank, id, y, x, angle, Pixels):
        self.PIX_CELL = Pixels
        self.parent = tank
        self.angle = angle
        self.done = False

        for (key, value) in zip(ammo_features, a_normal):
            self.__dict__[key] = value
        self.speed /= FRAME_RATE


        if angle > 1 or angle < -1:  # 0 - down, 90 - left, 180 - up, 270 - right. Or with minus
            angle /= 360
        self.speed_y = np.cos(-angle*np.pi*2) * self.speed
        self.speed_x = np.sin(-angle*np.pi*2) * self.speed
        self.Y = y + self.speed_y * 5
        self.X = x + self.speed_x * 5
        self.coords_yx = np.rint(np.array([self.Y, self.X])*self.PIX_CELL).astype(int)

        self.id_game = id
        self.tank_id = tank.id_game
        self.damage = 1.0  # 100%
        self.distance = 0
        self.damaged_target_id = 0
        self.damaged_target_yx = np.array([0,0])
        self.damage_dealed = 0
        self.target_YX = np.array([0, 0])  # for artillery_fly = True, with multy Pixels

        self.height = 0
        self.width = 0


    def __str__(self):
        atts = self.__dict__
        return 'Ammo\n'+'\n'.join([str(x) + ': ' + str(self.__dict__[x]) for x in atts])


    def move(self, coll_map):
        # remove old coords from collision map
        coll_map[round(self.Y*self.PIX_CELL), round(self.X*self.PIX_CELL), 2] = 0
        old_yx = copy(self.coords_yx)
        self.Y += self.speed_y
        self.X += self.speed_x
        self.coords_yx = np.rint(np.array([self.Y, self.X])*self.PIX_CELL).astype(int)
        self.distance += self.speed
        hit = False

        # hit obstacles
        if tar_id := coll_map[self.coords_yx[0], self.coords_yx[1], 0].sum() > 0:
            hit = True
            self.damaged_target_id = tar_id
            self.hit()

        # hit tanks
        if tar_id := coll_map[self.coords_yx[0], self.coords_yx[1], 1].sum() > 0:
            hit = True
            self.damaged_target_id = tar_id
            # TODO add tank armor calculation
            self.damage_dealed = self.hit()


        if self.artillery_fly:
            if (self.target_YX - self.coords_yx).sum() < 1:
                hit = True
                self.damage_dealed  = self.hit()

        if self.distance >= self.max_distance:
            self.done = True
            self.speed_y = 0
            self.speed_x = 0

        return np.rint(old_yx / self.PIX_CELL).astype(int), old_yx, hit


    def hit(self):
        # TODO add explosion
        self.done = True
        self.damaged_target_yx = self.coords_yx
        self.speed_y = 0
        self.speed_x = 0
        return max(((self.max_distance-self.distance)/self.max_distance) * self.parent.dmg, 0)


