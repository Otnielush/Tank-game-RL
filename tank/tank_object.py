import numpy as np
from video.graphics import FRAME_RATE, MOVES_PER_FRAME
from copy import copy

tank_type       = ['none', 'miner', 'freezer', 'artillery', 'laser', 'simple', 'tesla', 'repairer', 'heavy', 'base']
tank_features   = ['hp', 'dmg', 'reload_ammo', 'reload_skill', 'max_speed', 'speed_turn', 'speed_tower', 'ammo_type', 'armor_front', 'armor_side', 'armor_back', 'ammunition', 'sight_range']
t_simple        = [100,   20,       2,              5,             1,          20/360,          30/360,     'normal',       10,             7,           2,           50,           6]


class Tank():
    def __init__(self, id_game, player, y, x, Pix_Cell):
        # map placement
        self.X = x
        self.Y = y
        # TODO move this 2 to tank features
        self.width = 0.6
        self.height = 1

        self.crop_y = 0  # for coordinate placement on map -  0 + crop : X - crop
        self.crop_x = 0
        self.tank_coor_yx = []  # coordinates of tank for map placement and rotating
        self.PIX_CELL = Pix_Cell

        self.player = player
        self.name = player.name
        self.id_game = id_game
        self.id_player = player.id_connection
        self.type = player.tank_type
        self.speed = 0  # without Pixel multiplier
        self.speed_x = np.float(0)  # without Pixel multiplier
        self.speed_y = np.float(0)  # without Pixel multiplier

        self.direction_tank = 0  # Where body of tank looking. 0 - down, 90 - left, 180 - up, 270 - right. Or with minus
        self.direction_tower = 0  # 0 - same direction with body. More 0 - rotation right. Less 0 - rotation left
        self.reloading_ammo = 0  # seconds left
        self.reloading_skill = 0

        # Building attributes by tank type
        for (key, value) in zip(tank_features, t_simple):
            self.__dict__[key] = value

        self.max_speed /= FRAME_RATE
        self.speed_turn /= FRAME_RATE
        self.speed_tower /= FRAME_RATE
        self.reload_ammo *= FRAME_RATE
        self.reload_skill *= FRAME_RATE

        self.sight_mask = np.array([[1 if np.sqrt(x**2 + y**2) > self.sight_range else 0 for x in np.arange(-self.sight_range, self.sight_range+1, 1)] for y in np.arange(-self.sight_range, self.sight_range+1, 1)])
        self.crop_y = (Pix_Cell - self.height * Pix_Cell) / 2
        self.crop_x = (Pix_Cell - self.width * Pix_Cell) / 2
        # blue print of tank
        st_y = round((Pix_Cell*self.height) / 2)
        st_x = round((Pix_Cell*self.width) / 2)
        self.tank_coor_yx = np.meshgrid(np.arange(-st_y, (Pix_Cell*self.height - st_y)),
                                        np.arange(-st_x, (Pix_Cell*self.width - st_x)))
        # last placement of tank
        self.calc_tank_coordinates()



    def __str__(self):
        atts = self.__dict__
        return 'Tank\n'+'\n'.join([str(x)+': '+str(self.__dict__[x]) for x in atts])

    # return coordinates of rotated tank on map from 0
    def calc_tank_coordinates(self, y_pos=0, x_pos=0):
        yy = self.tank_coor_yx[1] * np.sin(np.pi * 2 * self.direction_tank) + self.tank_coor_yx[0] * -np.cos(np.pi * 2 * self.direction_tank)
        xx = self.tank_coor_yx[1] * np.cos(np.pi * 2 * self.direction_tank) + self.tank_coor_yx[0] * np.sin(np.pi * 2 * self.direction_tank)

        yy = yy + (self.crop_y + y_pos) - yy.min()
        xx = xx + (self.crop_x + x_pos - round(xx.min()/2))
        self.coords_yx = [np.rint(yy).astype(int), np.rint(xx).astype(int)]


    def calc_directions(self, turn_body, turn_tower):
        if turn_body > 1 or turn_body < -1:  # 0 - down, 90 - left, 180 - up, 270 - right. Or with minus
            turn_body /= 360
        if turn_tower > 1 or turn_tower < -1:
            turn_tower /= 360
        self.direction_tank += (turn_body * self.speed_turn)
        self.direction_tower += (turn_tower * self.speed_tower)

    def calc_speed_yx(self):
        self.speed_y = np.cos(-self.direction_tank*np.pi*2) * self.speed
        self.speed_x = np.sin(-self.direction_tank*np.pi*2) * self.speed


    # collision map layer, [accelerate {-1:1}, turn_body {-1:1}, turn_tower{-1:1}, shot (Boolean), skill (use, Boolean)]
    # accelerate - 0, turn_body - 1, turn_tower - 2, shot - 3, skill - 4
    # ! Return [old YX],old_coords[y[],x[]], shot, skill
    def move(self, coll_map, actions):
        # remove old coords from collision map
        coll_map[self.coords_yx[0], self.coords_yx[1], 1] = 0
        old_dir_tank = copy(self.direction_tank)
        old_yx = [copy(self.Y), copy(self.X)]
        old_coords = copy(self.coords_yx)

        self.calc_directions(actions[1], actions[2])
        self.speed += (self.max_speed * actions[0] * 0.2)  # 0.2 acceleration / brakes
        if self.speed > self.max_speed and self.speed > 0:
            self.speed = self.max_speed
        if self.speed < self.max_speed * 0.8 and self.speed < 0:
            self.speed = self.max_speed * -0.8

        self.calc_speed_yx()
        self.Y += self.speed_y
        self.X += self.speed_x

        self.calc_tank_coordinates(self.Y * self.PIX_CELL, self.X * self.PIX_CELL)

        # differences of moving
        diff_direction = old_dir_tank - self.direction_tank
        diff_y = self.Y - old_yx[0]
        diff_x = self.X - old_yx[1]

        check = 1
        while coll_map[self.coords_yx[0], self.coords_yx[1], :].sum() > 0:
            self.direction_tank -= (diff_direction / 4)
            self.Y -= diff_y / 4
            self.X -= diff_x / 4
            self.calc_tank_coordinates(self.Y * self.PIX_CELL, self.X * self.PIX_CELL)
            if check <= 4:
                check += 1
            else: break

        # Calculation for each move
        tick = 1
        if self.reloading_ammo > 0: self.reloading_ammo -= tick
        else: self.reloading_ammo = 0.0
        if self.reloading_skill > 0: self.reloading_skill -= tick
        else: self.reloading_skill = 0.0

        if actions[3] and self.reloading_ammo < 0.001:
            self.shot()
        else:
            actions[3] = False
        if actions[4] and self.reloading_skill < 0.001:
            self.use_skill()
        else:
            actions[4] = False


        return old_yx, old_coords, actions[3], actions[4]


    def shot(self):
        self.reloading_ammo = self.reload_ammo
        self.ammunition -= 1

    def use_skill(self):
        self.reloading_skill = self.reload_skill



# TODO: 1. Now creating only simple type of tank. Need to add more types.
#  2. Add flame tank




