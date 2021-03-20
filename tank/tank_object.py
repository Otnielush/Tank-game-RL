import numpy as np
from video.graphics import FRAME_RATE, MOVES_PER_FRAME

tank_type       = ['none', 'miner', 'freezer', 'artillery', 'laser', 'simple', 'tesla', 'repairer', 'heavy', 'base']
tank_features   = ['hp', 'dmg', 'reload_ammo', 'reload_skill', 'max_speed', 'speed_turn', 'speed_tower', 'ammo_type', 'armor_front', 'armor_side', 'armor_back', 'ammunition']
t_simple        = [100,   20,       2,              5,             1,          20,          30,         'normal',       10,             7,          2,              50]


class Tank():
    def __init__(self, id_game, player, x, y):
        self.player = player
        self.name = player.name
        self.X = x
        self.Y = y
        self.width = 1
        self.height = 1
        self.id_game = id_game
        self.id_player = player.id_connection
        self.type = player.tank_type
        self.speed = 0
        self.speed_x = np.float(0)
        self.speed_y = np.float(0)
        self.speed_YX = np.array([self.speed_y, self.speed_x], dtype=np.float)
        self.pos_YX = np.array([self.Y, self.X], dtype=np.float)
        self.direction_tank = 0  # Where body of tank looking. 0 - down, 90 - left, 180 - up, 270 - right. Or with minus
        self.direction_tower = 0  # 0 - same direction with body. More 0 - rotation right. Less 0 - rotation left
        self.reloading_ammo = 0  # seconds left
        self.reloading_skill = 0

        for (key, value) in zip(tank_features, t_simple):
            self.__dict__[key] = value

    def __str__(self):
        atts = self.__dict__
        return 'Tank\n'+'\n'.join([str(x)+': '+str(self.__dict__[x]) for x in atts])

    def calc_directions(self, turn_body, turn_tower):
        if turn_body > 1 or turn_body < -1:  # 0 - down, 90 - left, 180 - up, 270 - right. Or with minus
            turn_body /= 360
        if turn_tower > 1 or turn_tower < -1:
            turn_tower /= 360
        self.direction_tank += turn_body
        self.direction_tower += turn_tower

    def calc_speed_yx(self):
        self.speed_YX = [np.cos(self.direction_tank*np.pi*2) * self.speed, np.sin(self.direction_tank*np.pi*2) * self.speed]


    # collision map layer, accelerate {-1:1}, turn_body {-1:1}, turn_tower{-1:1}, shot (Boolean), skill (use, Boolean)
    # ! Return id, shot, skill
    def move(self, coll_map, accelerate, turn_body, turn_tower, shot, skill):
        self.calc_directions(turn_body, turn_tower)
        self.speed += (self.max_speed * accelerate * 0.2)  # 0.2 acceleration / brakes
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        self.calc_speed_yx()

        new_YX = self.pos_YX + self.speed_YX
        if coll_map[int(new_YX[0]), int(new_YX[1])] > 0:

            pass


        else:
            self.pos_YX = new_YX

        # Calculation for each move
        tick = (1/(FRAME_RATE*MOVES_PER_FRAME))
        if self.reloading_ammo > 0: self.reloading_ammo -= tick
        else: self.reloading_ammo = 0.0
        if self.reloading_skill > 0: self.reloading_skill -= tick
        else: self.reloading_skill = 0.0

        return self.id, shot, skill

    def shot(self):
        self.reloading_ammo = self.reload_ammo
        self.ammunition -= 1

    def use_skill(self):
        self.reloading_skill = self.reload_skill



# TODO: 1. Now creating only simple type of tank. Need to add more types.
#  2. Add flame tank
#  3. Finish move method




