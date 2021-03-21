import numpy as np
from video.graphics import FRAME_RATE, MOVES_PER_FRAME

tank_type       = ['none', 'miner', 'freezer', 'artillery', 'laser', 'simple', 'tesla', 'repairer', 'heavy', 'base']
tank_features   = ['hp', 'dmg', 'reload_ammo', 'reload_skill', 'max_speed', 'speed_turn', 'speed_tower', 'ammo_type', 'armor_front', 'armor_side', 'armor_back', 'ammunition', 'sight_range']
t_simple        = [100,   20,       2,              5,             1,          20,          30,         'normal',       10,             7,          2,              50,                6]


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

        self.sight_mask = np.array([[1 if np.sqrt(x**2 + y**2) > self.sight_range else 0 for x in np.arange(-self.sight_range, self.sight_range+1, 1)] for y in np.arange(-self.sight_range, self.sight_range+1, 1)])

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
        self.speed_y = np.cos(self.direction_tank*np.pi*2) * self.speed
        self.speed_x = np.sin(self.direction_tank*np.pi*2) * self.speed


    # collision map layer, [accelerate {-1:1}, turn_body {-1:1}, turn_tower{-1:1}, shot (Boolean), skill (use, Boolean)]
    # accelerate - 0, turn_body - 1, turn_tower - 2, shot - 3, skill - 4
    # ! Return id, [old YX], shot, skill
    def move(self, coll_map, actions):
        self.calc_directions(actions[1], actions[2])
        self.speed += (self.max_speed * actions[0] * 0.2)  # 0.2 acceleration / brakes
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        self.calc_speed_yx()

        old_yx = [self.Y, self.X]
        new_y = self.Y + self.speed_y
        new_x = self.X + self.speed_x
        if sum(coll_map[int(new_y), int(new_x), :]) > 0:

            pass


        else:
            self.Y = new_y
            self.X = new_x

        # Calculation for each move
        tick = (1/(FRAME_RATE*MOVES_PER_FRAME))
        if self.reloading_ammo > 0: self.reloading_ammo -= tick
        else: self.reloading_ammo = 0.0
        if self.reloading_skill > 0: self.reloading_skill -= tick
        else: self.reloading_skill = 0.0

        if actions[3] and self.reloading_ammo < 0.001:
            self.shot()
        if actions[4] and self.reloading_skill < 0.001:
            self.use_skill()

        return self.id_game, old_yx, actions[3], actions[4]

    def shot(self):
        self.reloading_ammo = self.reload_ammo
        self.ammunition -= 1

    def use_skill(self):
        self.reloading_skill = self.reload_skill



# TODO: 1. Now creating only simple type of tank. Need to add more types.
#  2. Add flame tank
#  3. Finish move method




