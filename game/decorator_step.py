from .game_object import TankGame
import numpy as np
from tank.ammo_object import Ammo
from scipy.ndimage.interpolation import rotate

def step(self):
    data = self.connection.get_actions()

    # Move Tanks
    #  coll_map, accelerate, turn_body, turn_tower, shot, skill
    for i in range(len(self.team1)):
        if self.team1[i].hp > 0:
            idd = self.team1[i].id_game
            idd, old_yx, old_coords, shot, skill = self.team1[i].move(self.map_coll, data[idd-self.ID_START])
            self.move_id_collision_map(self.team1[i], 1, old_yx, old_coords)  # changing collision map

            if shot:
                self.bullets.append(Ammo(self.team1[i], self.id_bul, self.team1[i].Y, self.team1[i].X, self.team1[i].direction_tank+self.team1[i].direction_tower))
                self.id_bul += 1  # id of bullets


    for i in range(len(self.team2)):
        if self.team2[i].hp > 0:
            idd = self.team2[i].id_game
            self.team2[i].move(self.map_coll, data[idd - self.ID_START])

    # Move Bullets
    # TODO: move bullets


    # TODO function after
    # self.calc_maps()
    return np.ones((self.height, self.width, 4))

setattr(TankGame, 'step', step)


def move_id_collision_map(self, obj, team, old_yx, old_coords):
    # TODO Stopped here. Rotate tank
    # self.map_coll[old_coords[0], old_coords[1], 1] = 0
    self.map_coll[obj.coords_yx[0], obj.coords_yx[1], 1] = obj.id_game
    self.map[old_coords[0], old_coords[1], team] = 0
    self.map[obj.coords_yx[0], obj.coords_yx[1], team] = self.tank_type_d[obj.type]
    self.map_env[round(old_yx[0]), round(old_yx[1]), team] = 0
    self.map_env[round(obj.Y): round(obj.Y + max(obj.height, 1)),
                round(obj.X): round(obj.X+max(obj.width, 1)), team] = self.tank_type_d[obj.type]




setattr(TankGame, 'move_id_collision_map', move_id_collision_map)