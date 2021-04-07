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

    # win check
    team1_alive = 0
    team2_alive = 0

    # Move Tanks
    #  coll_map, accelerate, turn_body, turn_tower, shot, skill
    for i in range(len(self.team1)):
        if self.team1[i].hp > 0:
            team1_alive += 1
            idd = self.team1[i].id_game - self.ID_START
            old_xy, old_coords, shot, skill = self.team1[i].move(self.map_coll, self.data[idd])
            self.move_obj_on_maps(self.team1[i], 1, old_xy, old_coords)  # changing maps
            self.reward(idd, self.score_move, 'move')

            if shot:
                # tank, id
                self.bullets.append(Ammo(self.team1[i], self.id_bul))
                self.bullets_in_act.append(self.id_bul-200,)
                self.id_bul += 1  # id of bullets
                self.reward(idd, self.score_shot, 'shot')


    for i in range(len(self.team2)):
        if self.team2[i].hp > 0:
            team2_alive += 1
            idd = self.team2[i].id_game - self.ID_START
            old_xy, old_coords, shot, skill = self.team2[i].move(self.map_coll, self.data[idd])
            self.move_obj_on_maps(self.team2[i], 2, old_xy, old_coords)  # changing maps
            self.reward(idd, self.score_move, 'move')

            if shot:
                self.bullets.append(Ammo(self.team2[i], self.id_bul))
                self.bullets_in_act.append(self.id_bul - 200,)
                self.id_bul += 1  # id of bullets
                self.reward(idd, self.score_shot, 'shot')


    # TODO Winning
    if team1_alive >= len(self.team1):
        pass
    elif team2_alive >= len(self.team2):
        pass


    # Move bullets
    for i in self.bullets_in_act:
        old_xy, old_coords, hit = self.bullets[i].move(self.map_coll)

        if hit:
            # check if tank damaged
            if self.bullets[i].damaged_target_id > 100:
                # move from array to new array
                self.bullets_in_act.remove(self.bullets[i].id_game - 200)
                # calculate angle of hitting target, angle 0 - down; 0.5 - up

                angle_diff = self.id_tanks[self.bullets[i].damaged_target_id].direction_tank - self.bullets[i].angle
                # front hit ±30°
                # TODO very simple hit place detection
                if abs(angle_diff) < 0.083:
                    side = 'back'
                # 30° - 150°
                elif angle_diff < -0.083 and angle_diff > -0.4167:
                    side = 'right'
                elif angle_diff > 0.083 and angle_diff < 0.4167:
                    side = 'left'
                else:
                    side = 'front'
                print('\nangle:', round(angle_diff,3),
                      'dir:',round(self.id_tanks[self.bullets[i].damaged_target_id].direction_tank,3),
                      'bul:', round(self.bullets[i].angle, 3))
                # calc dmg and hp to damaged tank
                dmg_dealed = self.id_tanks[self.bullets[i].damaged_target_id].damaged(self.bullets[i].damage_dealed_potencial, side)

                # give reward to shooter according to dmg and hit
                self.reward(self.bullets[i].parent.id_game - self.ID_START, self.score_hit+self.score_dmg*dmg_dealed, 'hit id: {} dmg: {}'.format(
                    str(int(self.bullets[i].damaged_target_id) - self.ID_START), dmg_dealed))
                # penalty to damaged tank
                self.reward(int(self.bullets[i].damaged_target_id) - self.ID_START,
                            self.score_take_hit + self.score_take_dmg * dmg_dealed, 'took dmg from id: {} dmg: {}'.format(
                        str(self.bullets[i].parent.id_game - self.ID_START), dmg_dealed))
                self.bullets[i].damaged_target_id

            # obstacles hit
            # if its wall -> broke or calc hp
            else:
                self.bullets_in_act.remove(self.bullets[i].id_game - 200)
                # if its wall and bullet type can destroy
                if self.bullets[i].damaged_target_id == self.map_obs_d['wall'] and self.bullets[i].destroy:
                    # erase wall from map
                    self.map[int(self.bullets[i].X)*self.PIX_CELL:int(self.bullets[i].X + 1)*self.PIX_CELL,
                                int(self.bullets[i].Y)*self.PIX_CELL:int(self.bullets[i].Y + 1)*self.PIX_CELL, 0] = 0
                    # map environment
                    self.map_env[int(self.bullets[i].X), int(self.bullets[i].Y), 0] = 0
                    # map collision
                    self.map_coll[int(self.bullets[i].X)*self.PIX_CELL:int(self.bullets[i].X + 1)*self.PIX_CELL,
                                int(self.bullets[i].Y)*self.PIX_CELL:int(self.bullets[i].Y + 1)*self.PIX_CELL, 0] = 0

            # erasing bullet from maps
            self.map[old_coords[0], old_coords[1], 3] = 0
            self.map_env[round(old_xy[0]), round(old_xy[1]), 3] = 0
        else:
            if self.bullets[i].done:
                # erasing from maps
                self.map[old_coords[0], old_coords[1], 3] = 0
                self.map_env[round(old_xy[0]), round(old_xy[1]), 3] = 0
                self.bullets_in_act.remove(self.bullets[i].id_game - 200)
            else:
                self.move_obj_on_maps(self.bullets[i], 3, old_xy, old_coords)


    self.steps += 1
    self.send_data_to_players()

setattr(TankGame, 'step', step)


def move_obj_on_maps(self, obj, layer, old_xy, old_coords):
    self.map_coll[obj.coords_xy[0], obj.coords_xy[1], 1 if layer < 3 else 2] = obj.id_game
    self.map[old_coords[0], old_coords[1], layer] = 0
    self.map[obj.coords_xy[0], obj.coords_xy[1], layer] = self.tank_type_d[obj.type] if layer < 3 else 1
    self.map_env[round(old_xy[0]), round(old_xy[1]), layer] = 0
    self.map_env[round(obj.X): round(obj.X+max(obj.width, 1)),
                    round(obj.Y): round(obj.Y + max(obj.height, 1)), layer] = self.tank_type_d[obj.type] if layer < 3 else 1


setattr(TankGame, 'move_obj_on_maps', move_obj_on_maps)