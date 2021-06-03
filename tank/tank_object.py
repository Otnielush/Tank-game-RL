import numpy as np
from options.video import FRAME_RATE, MOVES_PER_FRAME
from copy import copy

tank_type       = ['none', 'miner', 'freezer', 'artillery', 'laser', 'simple', 'tesla', 'repairer', 'heavy', 'base']
tank_features   = ['hp', 'dmg', 'reload_ammo', 'reload_skill', 'max_speed', 'speed_turn', 'speed_tower', 'ammo_type', 'armor_front', 'armor_side', 'armor_back', 'ammunition', 'sight_range']
t_simple        = [100,   30,       3,              5,             0.8,          30/360,          40/360,     'normal',       10,             7,           2,           40,           8]
t_laser         = [100,   20,       6,              5,             0.7,          30/360,          30/360,      'laser',       8,              4,           2,           30,           7]


class Tank():
    def __init__(self, id_game, player, team_num, Pix_Cell):
        # map placement
        self.X = 0
        self.Y = 0

        # TODO move this 2 to tank features
        self.width = 0.6
        self.height = 1

        self.crop_y = 0  # for coordinate placement on map -  0 + crop : X - crop
        self.crop_x = 0
        self.tank_coor_xy = []  # coordinates of tank for map placement and rotating
        self.PIX_CELL = Pix_Cell

        self.player = player
        self.name = player.name
        self.id_game = id_game
        self.team_num = team_num
        self.id_player = player.id_connection
        self.type = player.tank_type
        self.speed = 0  # without Pixel multiplier
        self.speed_x = np.float(0)  # without Pixel multiplier
        self.speed_y = np.float(0)  # without Pixel multiplier

        self.direction_tank = 0  # Where body of tank looking. 0 - down, 90 - left, 180 - up, 270 - right. Or with minus
        self.direction_tower = 0  # 0 - same direction with body. More 0 - rotation right. Less 0 - rotation left
        self.reloading_ammo = 0  # seconds left
        self.reloading_skill = 0
        self.capture_points = 0
        self.num_shots = 0
        self.num_hits = 0
        self.accuracy = 0  # percent

        # Building attributes by tank type
        tank_stats = globals()['t_'+self.type]
        for (key, value) in zip(tank_features, tank_stats):
            self.__dict__[key] = value

        self.max_speed /= FRAME_RATE
        self.speed_turn /= FRAME_RATE
        self.speed_tower /= FRAME_RATE
        self.reload_ammo *= FRAME_RATE
        self.reload_skill *= FRAME_RATE

        self.sight_mask = np.array([[1 if np.sqrt(x**2 + y**2) > self.sight_range else 0 for x in np.arange(-self.sight_range, self.sight_range+1, 1)] for y in np.arange(-self.sight_range, self.sight_range+1, 1)])
        self.crop_x = (Pix_Cell - self.width * Pix_Cell) / 2
        self.crop_y = (Pix_Cell - self.height * Pix_Cell) / 2
        # blue print of tank
        st_x = round((Pix_Cell*self.width) / 2)
        st_y = round((Pix_Cell*self.height) / 2)
        self.tank_coor_xy = np.meshgrid(np.arange(-st_x, (Pix_Cell*self.width - st_x)), np.arange(-st_y, (Pix_Cell * self.height - st_y)))
        # last placement of tank
        self.calc_tank_coordinates()

    def __str__(self):
        atts = self.__dict__
        return 'Tank\n'+'\n'.join([str(x)+': '+str(self.__dict__[x]) for x in atts])


    def rebuild(self):
        self.sight_mask = np.array([[1 if np.sqrt(x ** 2 + y ** 2) > self.sight_range else 0 for x in
                                     np.arange(-self.sight_range, self.sight_range + 1, 1)] for y in
                                    np.arange(-self.sight_range, self.sight_range + 1, 1)])

    # return coordinates of rotated tank on map from 0
    # TODO now when rotating coordinates is left top corner, need to change to center of object
    def calc_tank_coordinates(self,  x_pos=0, y_pos=0):
        xx = self.tank_coor_xy[0] * np.cos(np.pi * 2 * -self.direction_tank) - self.tank_coor_xy[1] * np.sin(np.pi * 2 * -self.direction_tank)
        yy = self.tank_coor_xy[0] * np.sin(np.pi * 2 * -self.direction_tank) + self.tank_coor_xy[1] * np.cos(np.pi * 2 * -self.direction_tank)

        xx = xx + (self.crop_x + x_pos - int(xx.min() / 2))
        yy = yy + (self.crop_y + y_pos) - yy.min()
        self.coords_xy = [xx.round().astype(int), yy.round().astype(int)]


    # turning tank body, turning tower
    def calc_directions(self, turn_body, turn_tower):
        if turn_body > 1 or turn_body < -1:  # 0 - down, 90 - left, 180 - up, 270 - right. Or with minus
            turn_body /= 360
        if turn_tower > 1 or turn_tower < -1:
            turn_tower /= 360
        self.direction_tank += (turn_body * self.speed_turn)
        self.direction_tower += (turn_tower * self.speed_tower)
        if self.direction_tank > 1:
            self.direction_tank -= 1
        elif self.direction_tank < 0:
            self.direction_tank += 1
        if self.direction_tower > 1:
            self.direction_tower -= 1
        elif self.direction_tower < 0:
            self.direction_tower += 1


    def calc_speed_XY(self, accelerate):
        self.speed += (self.max_speed * accelerate * 0.1)  # 0.1 acceleration (TODO add to features tank) / brakes
        if self.speed > self.max_speed and self.speed > 0:
            self.speed = copy(self.max_speed)
        if self.speed < self.max_speed * -0.8 and self.speed < 0:
            self.speed = self.max_speed * -0.8

        self.speed_x = np.sin(self.direction_tank*np.pi*2) * self.speed
        self.speed_y = np.cos(-self.direction_tank*np.pi*2) * self.speed

        self.X += self.speed_x
        self.Y += self.speed_y


    # collision map layer, [accelerate {-1:1}, turn_body {-1:1}, turn_tower{-1:1}, shot (Boolean), skill (use, Boolean)]
    # accelerate - 0, turn_body - 1, turn_tower - 2, shot - 3, skill - 4
    # ! Return [old XY],old_coords[x[],y[]], shot, skill
    def move(self, coll_map, actions):
        # remove old coords from collision map
        coll_map[self.coords_xy[0], self.coords_xy[1], 1] = 0

        old_xy = [copy(self.X), copy(self.Y)]
        old_coords = copy(self.coords_xy)
        old_speed = copy(self.speed)

        # turn
        old_dir_tank = copy(self.direction_tank)
        self.calc_directions(actions[1], actions[2])

        # speed
        self.calc_speed_XY(actions[0])
        speed_delta = self.speed - old_speed

        # coordinates
        self.calc_tank_coordinates(self.X * self.PIX_CELL, self.Y * self.PIX_CELL)

        # first check: turn + speed
        if coll_map[self.coords_xy[0], self.coords_xy[1], :].any() > 0:
            self.direction_tank = old_dir_tank
            self.speed -= speed_delta
            self.X = copy(old_xy[0])
            self.Y = copy(old_xy[1])
            self.coords_xy = copy(old_coords)

            # speed
            self.calc_speed_XY(actions[0])

            # coordinates
            self.calc_tank_coordinates(self.X * self.PIX_CELL, self.Y * self.PIX_CELL)

            # second check only speed
            if coll_map[self.coords_xy[0], self.coords_xy[1], :].any() > 0:
                self.speed = 0
                self.X = copy(old_xy[0])
                self.Y = copy(old_xy[1])
                self.coords_xy = copy(old_coords)


        # smart pushing tank away from obstacles


        # Calculation for each move
        tick = 1
        if self.reloading_ammo > 0: self.reloading_ammo -= tick
        else: self.reloading_ammo = 0.0
        if self.reloading_skill > 0: self.reloading_skill -= tick
        else: self.reloading_skill = 0.0

        if actions[3] and self.reloading_ammo < 0.001 and self.ammunition > 0:
            self.shot()
        else:
            actions[3] = False
        if actions[4] and self.reloading_skill < 0.001:
            self.use_skill()
        else:
            actions[4] = False

        return old_xy, old_coords, actions[3], actions[4]


    # dmg, side: 'front', 'left', 'right', 'back'
    def damaged(self, dmg, side):
        damage_dealed = 0

        # TODO add damage to NN -> weights of some neurons = 0
        # TODO add crit chances by damaged side
        if side == 'front':
            damage_dealed = dmg - self.armor_front
        elif side == 'left':
            damage_dealed = dmg - self.armor_side
        elif side == 'right':
            damage_dealed = dmg - self.armor_side
        elif side == 'back':
            damage_dealed = dmg - self.armor_back

        damage_dealed = round(min(max(damage_dealed, 0), self.hp))
        self.hp -= damage_dealed

        # death
        if self.hp <= 0:
            self.speed = 0
            self.speed_x = 0
            self.speed_y = 0
        # print('\ndamaged:', side, damage_dealed)
        return damage_dealed


    def shot(self):
        self.reloading_ammo = self.reload_ammo
        self.ammunition -= 1
        self.num_shots += 1

    def use_skill(self):
        self.reloading_skill = self.reload_skill



# TODO: 1. Now creating only simple type of tank. Need to add more types.
#  2. Add flame tank



