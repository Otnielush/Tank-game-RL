

tank_type       = ['none', 'miner', 'freezer', 'artillery', 'laser', 'simple', 'repairer', 'heavy', 'base']
tank_features   = ['hp', 'dmg', 'reload_ammo', 'reload_skill', 'speed', 'speed_turn', 'speed_tower', 'ammo_type', 'armor_front', 'armor_side', 'armor_back', 'ammunition']
t_simple        = [100,   20,       2,              5,             1,          20,          30,         'common',       10,             7,          2,              50]


class Tank():
    def __init__(self, id, tank_type):
        self.id = id
        self.type = tank_type
        self.speed = 0
        self.direction_tank = 0  # Where body of tank looking. 0 - up, 90 - right, 180 - down, 270 - left. Or with minus
        self.direction_tower = 0  # 0 - same direction with body. More 0 - rotation right. Less 0 - rotation left

        for (key, value) in zip(tank_features, t_simple):
            self.__dict__[key] = value

    def __str__(self):
        atts = self.__dict__
        return '\n'.join([str(x)+': '+str(self.__dict__[x]) for x in atts])


# TODO: Now creating only simple type of tank. Need to add more types




