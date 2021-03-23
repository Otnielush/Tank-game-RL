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
            old_yx, old_coords, shot, skill = self.team1[i].move(self.map_coll, data[idd-self.ID_START])
            self.move_obj_on_maps(self.team1[i], 1, old_yx, old_coords)  # changing maps

            if shot:
                # tank, id, y, x, angle, Pixels
                self.bullets.append(Ammo(self.team1[i], self.id_bul, self.team1[i].Y, self.team1[i].X,
                                         self.team1[i].direction_tank+self.team1[i].direction_tower, self.PIX_CELL))
                self.id_bul += 1  # id of bullets


    for i in range(len(self.team2)):
        if self.team2[i].hp > 0:
            idd = self.team2[i].id_game
            old_yx, old_coords, shot, skill = self.team2[i].move(self.map_coll, data[idd-self.ID_START])
            self.move_obj_on_maps(self.team2[i], 2, old_yx, old_coords)  # changing maps

            if shot:
                self.bullets.append(Ammo(self.team2[i], self.id_bul, self.team2[i].Y, self.team2[i].X,
                                         self.team2[i].direction_tank+self.team2[i].direction_tower, self.PIX_CELL))
                self.id_bul += 1  # id of bullets

    # Move bullets
    for i in range(len(self.bullets)):
        if not self.bullets[i].done:
            old_yx, old_coords, hit = self.bullets[i].move(self.map_coll)
            self.move_obj_on_maps(self.bullets[i], 3, old_yx, old_coords)
            # TODO ammo dmg
            if hit:
                pass


    self.send_data_to_players()


setattr(TankGame, 'step', step)


def move_obj_on_maps(self, obj, layer, old_yx, old_coords):
    self.map_coll[obj.coords_yx[0], obj.coords_yx[1], 1 if layer < 3 else 2] = obj.id_game
    self.map[old_coords[0], old_coords[1], layer] = 0
    self.map[obj.coords_yx[0], obj.coords_yx[1], layer] = self.tank_type_d[obj.type] if layer < 3 else 1
    self.map_env[round(old_yx[0]), round(old_yx[1]), layer] = 0
    self.map_env[round(obj.Y): round(obj.Y + max(obj.height, 1)),
                round(obj.X): round(obj.X+max(obj.width, 1)), layer] = self.tank_type_d[obj.type] if layer < 3 else 1




setattr(TankGame, 'move_obj_on_maps', move_obj_on_maps)