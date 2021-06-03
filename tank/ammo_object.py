import numpy as np
from copy import deepcopy
from options.video import FRAME_RATE


ammo_types = ['normal', 'rocket', 'bomb', 'jet', 'laser', 'electricity', 'mine', 'none']
# max_distance - deal 0 dmg, at start - 100%; destroy - can destroy obstacles; artillery_fly - fly over obstacles to target
# explossion - makes area dmg; expl_range - explossion range
ammo_features = ['speed', 'max_distance', 'destroy', 'artillery_fly', 'explossion', 'expl_range', 'life_time']  # distance - at start-max dmg, at end-0
a_normal      = [   4,          9,          True,       False,             False,           0,         0]
a_laser       = [ 4.2,          6,          False,      False,             False,           0,         1]


class Ammo():
    def __init__(self, tank, id):
        self.PIX_CELL = tank.PIX_CELL
        self.parent = tank
        self.type = tank.ammo_type
        self.angle = tank.direction_tank+tank.direction_tower
        if self.angle > 1:
            self.angle -= 1
        elif self.angle < 0:
            self.angle += 1

        self.done = False

        ammo_stats = globals()['a_'+self.type]
        for (key, value) in zip(ammo_features, ammo_stats):
            self.__dict__[key] = value
        self.speed /= FRAME_RATE
        self.life_time *= FRAME_RATE

        if self.angle > 1 or self.angle < -1:  # 0 - down, 90 - left, 180 - up, 270 - right. Or with minus
            self.angle /= 360
        self.speed_x = np.sin(self.angle*np.pi*2) * self.speed
        self.speed_y = np.cos(-self.angle*np.pi*2) * self.speed
        self.X = tank.X + tank.width/2 + self.speed_x * 3
        self.Y = tank.Y + tank.height/2 + self.speed_y * 3
        self.coords_xy = np.rint(np.array([self.X, self.Y])*self.PIX_CELL).astype(int)
        self.end_xy = deepcopy(self.coords_xy)/self.PIX_CELL

        self.damaged_target_id = 0
        self.damaged_target_yx = np.array([0, 0])
        self.distance = 0
        self.id_game = id
        self.tank_id = tank.id_game
        self.damage = 1.0  # 100%
        self.damage_dealed_potencial = 0
        self.target_XY = np.array([0, 0])  # for artillery_fly = True, with multy Pixels

        self.height = 0
        self.width = 0


    def __str__(self):
        atts = self.__dict__
        return 'Ammo\n'+'\n'.join([str(x) + ': ' + str(self.__dict__[x]) for x in atts])


    def move(self, coll_map):
        # laser life time
        if self.type == 'laser':
            if self.distance == 0:  # make shot
                hit = False
                self.distance = self.max_distance
                angle_x = np.sin(self.angle*np.pi*2)
                angle_y = np.cos(-self.angle*np.pi*2)
                coords_x = (np.arange(0, self.max_distance, 0.2) * angle_x * self.PIX_CELL + (self.X * self.PIX_CELL)).round().astype(int)
                coords_y = (np.arange(0, self.max_distance, 0.2) * angle_y * self.PIX_CELL + (self.Y * self.PIX_CELL)).round().astype(int)

                self.end_xy[0] = deepcopy(coords_x[-1]) / self.PIX_CELL
                self.end_xy[1] = deepcopy(coords_y[-1]) / self.PIX_CELL
                # hit check
                for i in range(len(coords_x)):
                    tar_id_tank = coll_map[coords_x[i], coords_y[i], 1]
                    tar_id_obs = coll_map[coords_x[i], coords_y[i], 0]
                    if tar_id_tank.any() > 0 or tar_id_obs.any() > 0:
                        if tar_id_tank.any() > 0:
                            self.parent.num_hits += 1
                        hit = True
                        self.damaged_target_id = max(tar_id_tank.max(), tar_id_obs.max())
                        self.damaged_target_xy = deepcopy([coords_x[i], coords_y[i]])
                        self.distance = self.max_distance * i/len(coords_x) + 0.01
                        self.damage_dealed_potencial = ((self.max_distance-self.distance/3)/self.max_distance) * self.parent.dmg
                        self.end_xy[0] = deepcopy(coords_x[i]) / self.PIX_CELL
                        self.end_xy[1] = deepcopy(coords_y[i]) / self.PIX_CELL
                        self.done = False
                        break
                return np.array([self.X, self.Y]), self.coords_xy, hit
            else:
                if self.life_time > 0:  # shot but on screen
                    self.life_time -= 1
                else:
                    self.done = True
                return np.array([self.X, self.Y]), self.coords_xy, False

        # remove old coords from collision map
        coll_map[int(round(self.X*self.PIX_CELL)), int(round(self.Y*self.PIX_CELL)), 2] = 0
        old_xy = deepcopy(self.coords_xy)
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
            self.parent.num_hits += 1
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

        return (old_xy / self.PIX_CELL).round().astype(int), old_xy, hit


    def hit(self):
        # TODO add explosion
        self.done = True
        self.damaged_target_xy = self.coords_xy
        self.speed_x = 0
        self.speed_y = 0
        self.speed = 0
        self.damage_dealed_potencial = max(((self.max_distance-self.distance)/self.max_distance) * self.parent.dmg, 0)


