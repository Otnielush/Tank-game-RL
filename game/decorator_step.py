from .game_object import TankGame
from tank.ammo_object import Ammo
import time
from options.video import MOVES_PER_FRAME, WIDTH, HEIGHT, FRAME_RATE
from video.graphics import destroy_tank, destroy_wall
from pygame.transform import rotate, scale

Capture_points_win = 40*FRAME_RATE


# calculate moves from actions
# observation, reward, done, info = env.step(actions)  - must be like that
def step(self):
    info = None
    done = False
    win_draw = False
    # MOVES_PER_FRAME mechanics. second part of function is at the end of function
    if self.frame_step <= 0:
        self.data = self.connection.get_actions()


    # Win check
    team1_alive = 0
    team2_alive = 0
    team1_capture_points = 0
    team2_capture_points = 0

    # Move Tanks
    #  coll_map, accelerate, turn_body, turn_tower, shot, skill
    for i in range(len(self.team1)):
        if self.team1[i].hp > 0:
            team1_alive += 1
            idd = self.team1[i].id_game - self.ID_START
            old_xy, old_coords, shot, skill = self.team1[i].move(self.map_coll, self.data[idd])
            self.move_obj_on_maps(self.team1[i], 1, old_xy, old_coords)  # changing maps
            self.reward(idd, self.score_move * self.team1[i].speed, 'move')
            # capture points
            if (((self.team1[i].coords_xy[0] >= self.map_base_xy[1, 0, 0]) & (self.team1[i].coords_xy[0] <= self.map_base_xy[1, 0, 1]))
            & ((self.team1[i].coords_xy[1] >= self.map_base_xy[1, 1, 0]) & (self.team1[i].coords_xy[1] <= self.map_base_xy[1, 1, 1]))).any():
                self.team1[i].capture_points += 1
                team1_capture_points += self.team1[i].capture_points
            else:
                self.team1[i].capture_points = 0

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
            self.reward(idd, self.score_move * self.team2[i].speed, 'move')
            # capture points
            if (((self.team2[i].coords_xy[0] >= self.map_base_xy[0, 0, 0]) & (self.team2[i].coords_xy[0] <= self.map_base_xy[0, 0, 1]))
                & ((self.team2[i].coords_xy[1] >= self.map_base_xy[0, 1, 0]) & (self.team2[i].coords_xy[1] <= self.map_base_xy[0, 1, 1]))).any():
                self.team2[i].capture_points += 1
                team2_capture_points += self.team2[i].capture_points
            else:
                self.team2[i].capture_points = 0

            if shot:
                self.bullets.append(Ammo(self.team2[i], self.id_bul))
                self.bullets_in_act.append(self.id_bul - 200,)
                self.id_bul += 1  # id of bullets
                self.reward(idd, self.score_shot, 'shot')

    # print capturing points TODO add to input to NN
    if team1_capture_points > 0 or team2_capture_points > 0:
        print('Capturing:', team1_capture_points, team2_capture_points, end=' |')

    # Move bullets
    for i in self.bullets_in_act:
        old_xy, old_coords, hit = self.bullets[i].move(self.map_coll)

        if hit:
            # laser life time check
            if self.bullets[i].life_time <= 1:
                self.bullets_in_act.remove(self.bullets[i].id_game - 200)
            # check if tank damaged
            if self.bullets[i].damaged_target_id > 100:
                # move from array to new array

                # calculate angle of hitting target, angle 0 - down; 0.5 - up

                angle_diff = self.id_tanks[self.bullets[i].damaged_target_id].direction_tank - self.bullets[i].angle
                # front hit ??30??
                # TODO very simple hit place detection
                if abs(angle_diff) < 0.083:
                    side = 'back'
                # 30?? - 150??
                elif angle_diff < -0.083 and angle_diff > -0.4167:
                    side = 'right'
                elif angle_diff > 0.083 and angle_diff < 0.4167:
                    side = 'left'
                else:
                    side = 'front'

                # calc dmg and hp to damaged tank
                dmg_dealed = self.id_tanks[self.bullets[i].damaged_target_id].damaged(self.bullets[i].damage_dealed_potencial, side)

                # team of damaged tank
                tank_dmg_t = self.id_tanks[self.bullets[i].damaged_target_id].team_num
                # team of fired tank
                tank_frd_t = self.bullets[i].parent.team_num

                # tank destroyed
                if self.id_tanks[self.bullets[i].damaged_target_id].hp <= 0:
                    # remove from maps
                    self.map_env[round(self.id_tanks[self.bullets[i].damaged_target_id].X),
                                 round(self.id_tanks[self.bullets[i].damaged_target_id].Y), tank_dmg_t] = 0
                    self.map_coll[self.id_tanks[self.bullets[i].damaged_target_id].coords_xy[0],
                                  self.id_tanks[self.bullets[i].damaged_target_id].coords_xy[1], 1] = 0
                    # video
                    if self.VIDEO[0]:
                        destroy_tank(self.id_tanks[self.bullets[i].damaged_target_id].X, self.id_tanks[self.bullets[i].damaged_target_id].Y,
                                     self.id_tanks[self.bullets[i].damaged_target_id].width, self.id_tanks[self.bullets[i].damaged_target_id].height,
                                     self.id_tanks[self.bullets[i].damaged_target_id].direction_tank)
                        # tank_body = rotate(scale(tank_destroyed, (round(self.id_tanks[self.bullets[i].damaged_target_id].width*MULTY_PIXEL_V),
                        #                                           round(self.id_tanks[self.bullets[i].damaged_target_id].height*MULTY_PIXEL_V))),
                        #                    self.id_tanks[self.bullets[i].damaged_target_id].direction_tank * 360 + 180)
                        # background.blit(tank_body, (self.id_tanks[self.bullets[i].damaged_target_id].X * MULTY_PIXEL_V,
                        #                             self.id_tanks[self.bullets[i].damaged_target_id].Y * MULTY_PIXEL_V))
                    # stats
                    self.id_tanks[self.bullets[i].damaged_target_id].player.deaths += 1
                    self.bullets[i].parent.player.tanks_killed += 1
                    # rewards
                    self.reward(self.bullets[i].parent.id_game - self.ID_START,
                                self.score_kill, 'killed id: {}'.format(str(int(self.bullets[i].damaged_target_id) - self.ID_START)))
                    self.reward(int(self.bullets[i].damaged_target_id) - self.ID_START,
                                self.score_death, 'killed by id: {}'.format(str(self.bullets[i].parent.id_game - self.ID_START)))


                # friendly fire check
                # if not FF
                if tank_frd_t != tank_dmg_t:
                    # give reward to shooter according to dmg and hit
                    self.reward(self.bullets[i].parent.id_game - self.ID_START, self.score_hit+self.score_dmg*dmg_dealed, 'hit id: {} dmg: {}'.format(
                        str(int(self.bullets[i].damaged_target_id) - self.ID_START), dmg_dealed))
                    # penalty to damaged tank
                    self.reward(int(self.bullets[i].damaged_target_id) - self.ID_START,
                                self.score_take_hit + self.score_take_dmg * dmg_dealed, 'took dmg from id: {} dmg: {}'.format(
                            str(self.bullets[i].parent.id_game - self.ID_START), dmg_dealed))
                # friendly fire
                else:
                    # give penalty to shooter according to dmg and hit
                    self.reward(self.bullets[i].parent.id_game - self.ID_START,
                                self.score_friendly_fire * self.score_dmg * dmg_dealed, 'hit friendly id: {} dmg: {}'.format(
                            str(int(self.bullets[i].damaged_target_id) - self.ID_START), dmg_dealed))
                    # penalty to damaged tank
                    self.reward(int(self.bullets[i].damaged_target_id) - self.ID_START,
                                self.score_take_hit + self.score_take_dmg * dmg_dealed,
                                'took friendly dmg from id: {} dmg: {}'.format(
                                    str(self.bullets[i].parent.id_game - self.ID_START), dmg_dealed))

                # Remove capture points from damaged tank
                if self.id_tanks[self.bullets[i].damaged_target_id].capture_points > 0:
                    # reward
                    self.reward(self.bullets[i].parent.id_game - self.ID_START,
                                self.score_capture, 'stopped capturing id: {} points: {}'.format(
                                    str(int(self.bullets[i].damaged_target_id) - self.ID_START),
                            self.id_tanks[self.bullets[i].damaged_target_id].capture_points))
                    self.id_tanks[self.bullets[i].damaged_target_id].capture_points = 0



            # obstacles hit
            # if its wall -> broke or calc hp
            else:
                # if its wall and bullet type can destroy
                if self.bullets[i].damaged_target_id == self.map_obs_d['wall'] and self.bullets[i].destroy:
                    # erase wall from map
                    # self.map[int(self.bullets[i].X)*self.PIX_CELL:int(self.bullets[i].X + 1)*self.PIX_CELL,
                    #             int(self.bullets[i].Y)*self.PIX_CELL:int(self.bullets[i].Y + 1)*self.PIX_CELL, 0] = 0
                    # map environment
                    self.map_env[int(self.bullets[i].X), int(self.bullets[i].Y), 0] = 0
                    # map collision
                    self.map_coll[int(self.bullets[i].X)*self.PIX_CELL:int(self.bullets[i].X + 1)*self.PIX_CELL,
                                int(self.bullets[i].Y)*self.PIX_CELL:int(self.bullets[i].Y + 1)*self.PIX_CELL, 0] = 0
                    if self.VIDEO[0]:
                        destroy_wall(self.bullets[i].X, self.bullets[i].Y)
                        # background.blit(land, (int(self.bullets[i].X) * MULTY_PIXEL_V, int(self.bullets[i].Y) * MULTY_PIXEL_V))

            # erasing bullet from maps
            try:
                self.map_env[int(old_xy[0]), int(old_xy[1]), 3] = 0
            except:
                print('problem bullet', self.bullets[i])

        else:
            if self.bullets[i].done:
                # erasing from maps
                # self.map[old_coords[0], old_coords[1], 3] = 0
                self.map_env[round(old_xy[0]), round(old_xy[1]), 3] = 0
                self.bullets_in_act.remove(self.bullets[i].id_game - 200)
            else:
                self.move_obj_on_maps(self.bullets[i], 3, old_xy, old_coords)


    # TODO info results of match
    # team 2 WIN
    # eluminate all team; capture base
    if team1_alive <= 0 or team2_capture_points > Capture_points_win:
        if team2_alive <= 0:
            win_draw = True
        else:
            info = {'game_done': True}
            done = True
            # rewards
            for t in self.team2:
                self.reward(t.id_game - self.ID_START, t.capture_points * self.score_capture, 'for capture')
                self.reward(t.id_game - self.ID_START, self.score_win, 'win')
                if team2_capture_points > Capture_points_win and t.capture_points > 0:
                   t.player.bases_captured += 1
                else:
                    t.player.wins += 1
            for t in self.team1:
                self.reward(t.id_game - self.ID_START, self.score_lose, 'lose')
    # Team 1 WIN
    elif team2_alive <= 0 or team1_capture_points > Capture_points_win:
        info = {'game_done': True}
        done = True
        # rewards
        for t in self.team1:
            self.reward(t.id_game - self.ID_START, t.capture_points*self.score_capture, 'for capture')
            self.reward(t.id_game - self.ID_START, self.score_win, 'win')
            if team1_capture_points > Capture_points_win and t.capture_points > 0:
                t.player.bases_captured += 1
            else:
                t.player.wins += 1
        for t in self.team2:
            self.reward(t.id_game - self.ID_START, self.score_lose, 'lose')

    # Draw: all dead or time gone
    if win_draw or self.time_passed >= self.time_round_len:
        info = {'game_done': True}
        done = True
        # rewards
        for t in self.team1:
            self.reward(t.id_game - self.ID_START, t.capture_points*self.score_capture, 'for capture')
            self.reward(t.id_game - self.ID_START, self.score_win + self.score_lose, 'draw')
            t.player.draws += 1
        for t in self.team2:
            self.reward(t.id_game - self.ID_START, t.capture_points * self.score_capture, 'for capture')
            self.reward(t.id_game - self.ID_START, self.score_win + self.score_lose, 'draw')
            t.player.draws += 1


    # MOVES_PER_FRAME mechanics. first part of function is at the start of function
    if MOVES_PER_FRAME > 1 or done:
        if self.frame_step == 1 or done:
            self.send_data_to_players(info)
            self.frame_step -= 1
        elif self.frame_step <= 0:
            self.frame_step = MOVES_PER_FRAME - 1
        else:
            self.frame_step -= 1
    else:
        self.send_data_to_players(info)


    # print('time:', round(timer), end='')
    self.time_passed += 1
    return done

setattr(TankGame, 'step', step)



def move_obj_on_maps(self, obj, layer, old_xy, old_coords):
    self.map_coll[obj.coords_xy[0], obj.coords_xy[1], 1 if layer < 3 else 2] = obj.id_game
    # self.map[old_coords[0], old_coords[1], layer] = 0
    # self.map[obj.coords_xy[0], obj.coords_xy[1], layer] = self.tank_type_d[obj.type] if layer < 3 else 1
    self.map_env[int(round(old_xy[0])), int(round(old_xy[1])), layer] = 0
    self.map_env[int(round(obj.X)): int(round(obj.X+max(obj.width, 1))),
                    int(round(obj.Y)): int(round(obj.Y + max(obj.height, 1))), layer] = self.tank_type_d[obj.type] if layer < 3 else 1


setattr(TankGame, 'move_obj_on_maps', move_obj_on_maps)