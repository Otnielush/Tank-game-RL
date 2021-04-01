from .game_object import TankGame
import numpy as np
from tank.ammo_object import Ammo
from options.video import MOVES_PER_FRAME

# calculate moves from actions
def step(self):
    if self.frame_step <= 0:
        self.data = self.connection.get_actions()
        self.frame_step = MOVES_PER_FRAME - 1

        if MOVES_PER_FRAME > 1:
            self.frame_step -= 1

    # Move Tanks
    #  coll_map, accelerate, turn_body, turn_tower, shot, skill
    for i in range(len(self.team1)):
        if self.team1[i].hp > 0:
            idd = self.team1[i].id_game
            old_xy, old_coords, shot, skill = self.team1[i].move(self.map_coll, self.data[idd - self.ID_START])
            self.move_obj_on_maps(self.team1[i], 1, old_xy, old_coords)  # changing maps

            if shot:
                # tank, id
                self.bullets.append(Ammo(self.team1[i], self.id_bul))
                self.bullets_in_act += (self.id_bul-200,)
                self.id_bul += 1  # id of bullets


    for i in range(len(self.team2)):
        if self.team2[i].hp > 0:
            idd = self.team2[i].id_game
            old_xy, old_coords, shot, skill = self.team2[i].move(self.map_coll, self.data[idd - self.ID_START])
            self.move_obj_on_maps(self.team2[i], 2, old_xy, old_coords)  # changing maps

            if shot:
                self.bullets.append(Ammo(self.team2[i], self.id_bul))
                self.bullets_in_act += (self.id_bul - 200,)
                self.id_bul += 1  # id of bullets

    # Move bullets
    for i in range(len(self.bullets)):
        if not self.bullets[i].done:
            old_xy, old_coords, hit = self.bullets[i].move(self.map_coll)
            if hit:
                self.rewards[self.bullets[i].damaged_target_id]
                self.bullets[i].damaged_target_id
            self.move_obj_on_maps(self.bullets[i], 3, old_xy, old_coords)
            # TODO ammo dmg
            if hit:
                pass


    self.send_data_to_players()


setattr(TankGame, 'step', step)


def move_obj_on_maps(self, obj, layer, old_xy, old_coords):
    self.map_coll[obj.coords_xy[0], obj.coords_xy[1], 1 if layer < 3 else 2] = obj.id_game
    self.map[old_coords[0], old_coords[1], layer] = 0
    self.map[obj.coords_xy[0], obj.coords_xy[1], layer] = self.tank_type_d[obj.type] if layer < 3 else 1
    self.map_env[int(old_xy[0]), int(old_xy[1]), layer] = 0
    self.map_env[int(obj.X): int(obj.X+max(obj.width, 1)),
                    int(obj.Y): int(obj.Y + max(obj.height, 1)), layer] = self.tank_type_d[obj.type] if layer < 3 else 1


setattr(TankGame, 'move_obj_on_maps', move_obj_on_maps)