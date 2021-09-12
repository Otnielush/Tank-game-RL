from os import path, mkdir

import numpy as np
import random
from copy import copy
from tank.tank_object import tank_type, Tank
from net.broadcasting import net_connection
from options.video import FRAME_RATE
import pandas as pd
from player.player_RL import TEAM_SPIRIT

# class for training
class dummy():
    def __init__(self, moving=False, tank_type='simple'):
        self.name = 'dummy'
        self.id_connection = 100000
        self.tank_type = tank_type
        self.games_played = 0
        self.wins = 0
        self.deaths = 0
        self.id_game = 0
        self.connection = 0
        self.env = 0
        self.data = 0
        self.reward = 0
        self.info = 0
        self.tank_ingame = ''  # tank object in game
        self.start_side = ''  # starting side of map for RL

        self.games_played = 0
        self.wins = 0
        self.tanks_killed = 0
        self.deaths = 0
        self.bases_captured = 0
        self.draws = 0
        self.num_shots = 0
        self.num_hits = 0

        self.moving = moving


    def change_id(self, id_game):
        self.id_game = id_game

    def connected_new_game(self, net_connection):
        self.connection = net_connection

    def move(self):
        if self.moving:
            self.connection.send_action(self.id_game, [(random.random()-0.5)/2, (random.random()-0.5)/2, (random.random()-0.5)/2, 0, 0])
    def done(self):
        pass

# game object
# Steps: create, new_game
class TankGame():

    def __init__(self, pixels=5):
        self.PIX_CELL = pixels
        self.team_size = 0
        self.width = 0
        self.height = 0
        self.map_obs = ['land', 'bush', 'desert', 'forest', 'water', 'swamp', 'wall', 'rock']
        self.map_obs_d = {'land': 0, 'bush': 0.14, 'desert': 0.29, 'forest': 0.43, 'water': 0.57, 'swamp': 0.71, 'wall': 0.86, 'rock': 1}  # Dictionary for obstacles
        self.tank_type_d = {t[0]:t[1] for t in zip(tank_type, np.linspace(0, 1, len(tank_type)))}
        self.team1 = []  # [Tank]
        self.team2 = []
        self.id_tanks = {}
        self.connection = 0
        self.rewards = 0  # [id, step] # new game generates
        self.rewards_comment = 0
        self.steps = 0

        # SCORES
        self.score_win              = 10
        self.score_lose             = -10
        self.score_hit              = 0.5
        self.score_kill             = 3
        self.score_dmg              = 0.3
        self.score_death            = -5
        self.score_kill_assist      = 1
        self.score_exploring        = 1
        self.score_friendly_fire    = -3
        self.score_move             = -0.3
        self.score_shot             = -0.35
        self.score_take_hit         = self.score_hit * -0.5
        self.score_take_dmg         = self.score_dmg * -1
        self.score_capture          = 0.01

        self.data = 0
        self.frame_step = 0

        # training
        self.shooting_moving = False


    # start new round with array of connected players [player, ...]
    # id starts from 101
    def new_game(self, width, height, team1_pl, team2_pl, VIDEO, type_m='random'):
        print('\nNew game, type:', type_m)
        self.width = width
        self.height = height
        self.VIDEO = VIDEO

        self.steps = 0
        self.frame_step = 0
        self.team1 = []  # [Tank]
        self.team2 = []
        self.team1_ids = []
        self.team2_ids = []
        self.id_tanks = {}
        self.game_type = type_m
        self.ID_START = 101
        id_tank = self.ID_START
        self.bullets = []
        self.bullets_in_act = []  # ids of bullets - id_bul ( for bullets array )
        self.id_bul = 200   # id bullets
        self.time_round_len = 5 * 60 * FRAME_RATE  # frames


        if type_m == 'shooting':
            self.height = 15
            self.width = 15
            num_players_train = len(team1_pl)
            team1_players = np.concatenate((team1_pl, [dummy() for _ in range(4)]), axis=None)
            team2_players = [dummy(moving=self.shooting_moving) for _ in range(6)]
            self.time_round_len = int(self.time_round_len * 0.5)
        else:
            team1_players = copy(team1_pl)
            team2_players = copy(team2_pl)
        del(team1_pl, team2_pl)

        num_players = len(team1_players) + len(team2_players)
        self.connection = net_connection(num_players, False, (self.width, self.height, 4), 10, (5,))

        # Creating object Tank for players and sending connection
        for i in range(len(team1_players)):
            self.team1.append(Tank(id_tank, team1_players[i], 1, self.PIX_CELL))
            team1_players[i].change_id(id_game=id_tank)
            team1_players[i].tank_ingame = self.team1[-1]
            team1_players[i].games_played += 1
            self.team1[i].player.connected_new_game(self.connection)
            self.id_tanks[id_tank] = self.team1[-1]
            self.team1_ids.append(id_tank-self.ID_START)
            id_tank += 1
        for i in range(len(team2_players)):
            self.team2.append(Tank(id_tank, team2_players[i], 2, self.PIX_CELL))
            team2_players[i].change_id(id_game=id_tank)
            team2_players[i].tank_ingame = self.team2[-1]
            team2_players[i].games_played += 1
            self.team2[i].player.connected_new_game(self.connection)
            self.id_tanks[id_tank] = self.team2[-1]
            self.team2_ids.append(id_tank - self.ID_START)
            id_tank += 1

        # (num_players, self.time_round_len, 2))  # {id, step, [reward, comment]}
        # self.rewards = [[[0, '']
        #                  for _ in range(self.time_round_len)]
        #                 for _ in range(num_players)]
        self.rewards = np.zeros((num_players, self.time_round_len+5))
        self.rewards_comment = np.zeros((num_players, self.time_round_len+5), dtype='<U30')

        # attributes by game type
        if type_m == 'shooting':
            for tank in self.team1:
                tank.reload_ammo /= 4
                if tank.name == 'dummy':
                    tank.hp = 20
                else:
                    tank.sight_range = 20
                    tank.ammunition *= 2
                    tank.rebuild()
            for tank in self.team2:
                tank.hp = 20
            self.map_generate_shooting(num_players_train)
        else:
            self.map_generate_random()
        # Round timer
        self.time_passed = 0
        # sending ENV to network connection
        self.send_data_to_players({'game_start': True})


    def send_data_to_players(self, info):
        timer = (self.time_round_len - self.time_passed) / self.time_round_len
        env_team1 = self.build_env_map_team(1)
        for i in range(len(self.team1)):
            if self.team1[i].name == 'dummy':
                continue
            idd = self.team1[i].id_game - self.ID_START

            # env_map
            # 10: x, y, angle_tank, angle_tower, hp, speed, (time to reload: ammo, skill); ammunition; round time left in %;
            # reward; info(start game, game done);
            self.connection.send_env_to_players(idd, env_team1,
                    [self.team1[i].X, self.team1[i].Y, self.team1[i].direction_tank, self.team1[i].direction_tower, self.team1[i].hp,
                     self.team1[i].speed, self.team1[i].reloading_ammo, self.team1[i].reloading_skill, self.team1[i].ammunition, timer],
                    copy(self.rewards[idd, self.steps]), info)
        env_team2 = self.build_env_map_team(2)
        for i in range(len(self.team2)):
            if self.team2[i].name == 'dummy':
                continue
            idd = self.team2[i].id_game - self.ID_START
            self.connection.send_env_to_players(idd, env_team2,
                    [self.team2[i].X, self.team2[i].Y, self.team2[i].direction_tank, self.team2[i].direction_tower, self.team2[i].hp,
                     self.team2[i].speed, self.team2[i].reloading_ammo, self.team2[i].reloading_skill, self.team2[i].ammunition, timer],
                    copy(self.rewards[idd, self.steps]), info)
        self.steps += 1


    # Making input for players
    def build_env_map_team(self, team_num):
        if team_num == 1:
            map_env = copy(self.map_env[:, :, :4])
            mask = np.ones(map_env.shape[:2])
            for i in range(len(self.team1)):
                if self.team1[i].name == 'dummy' or self.team1[i].hp <= 0:
                    continue
                start_x = max(int(self.team1[i].X - self.team1[i].sight_range), 0)
                start_y = max(int(self.team1[i].Y - self.team1[i].sight_range), 0)
                start_x_m = max(int(self.team1[i].sight_range - self.team1[i].X), 0)
                start_y_m = max(int(self.team1[i].sight_range - self.team1[i].Y), 0)

                end_x = min(int(self.team1[i].X + self.team1[i].sight_range + 1), self.width)
                end_y = min(int(self.team1[i].Y + self.team1[i].sight_range + 1), self.height)
                end_x_m = min(self.team1[i].sight_range * 2 + 1 - start_x_m, self.team1[i].sight_range * 2 + 1)
                end_y_m = min(self.team1[i].sight_range * 2 + 1 - start_y_m, self.team1[i].sight_range * 2 + 1)

                if end_x - start_x < end_x_m - start_x_m:
                    end_x_m = start_x_m + (end_x - start_x)
                elif end_x - start_x > end_x_m - start_x_m:
                    end_x = start_x + (end_x_m - start_x_m)
                if end_y - start_y < end_y_m - start_y_m:
                    end_y_m = start_y_m + (end_y - start_y)
                elif end_y - start_y > end_y_m - start_y_m:
                    end_y = start_y + (end_y_m - start_y_m)

                mm = self.team1[i].sight_mask[start_x_m: end_x_m, start_y_m: end_y_m]
                mask[start_x: end_x, start_y: end_y] *= mm

            mask = (mask - 1) * -1
            map_env[:, :, 2] *= mask
            return map_env

        else:
            map_env = copy(self.map_env[:, :, [0, 2, 1, 3]])   # Layers: 1- friendly team, 2 - enemy`s team
            mask = np.ones(map_env.shape[:2])
            for i in range(len(self.team2)):
                if self.team2[i].name == 'dummy' or self.team2[i].hp <= 0:
                    continue
                start_x = max(int(self.team2[i].X - self.team2[i].sight_range), 0)
                start_y = max(int(self.team2[i].Y - self.team2[i].sight_range), 0)
                start_x_m = max(int(self.team2[i].sight_range - self.team2[i].X), 0)
                start_y_m = max(int(self.team2[i].sight_range - self.team2[i].Y), 0)

                # TODO rebuild to ranges like in player_RL
                end_x = min(int(self.team2[i].X + self.team2[i].sight_range + 1), self.width)
                end_y = min(int(self.team2[i].Y + self.team2[i].sight_range + 1), self.height)
                end_x_m = min(self.team2[i].sight_range * 2 + 1 - start_x_m, self.team2[i].sight_range * 2 + 1)
                end_y_m = min(self.team2[i].sight_range * 2 + 1 - start_y_m, self.team2[i].sight_range * 2 + 1)

                if end_x - start_x < end_x_m - start_x_m:
                    end_x_m = start_x_m + (end_x - start_x)
                elif end_x - start_x > end_x_m - start_x_m:
                    end_x = start_x + (end_x_m - start_x_m)
                if end_y - start_y < end_y_m - start_y_m:
                    end_y_m = start_y_m + (end_y - start_y)
                elif end_y - start_y > end_y_m - start_y_m:
                    end_y = start_y + (end_y_m - start_y_m)

                mm = self.team2[i].sight_mask[start_x_m: end_x_m, start_y_m: end_y_m]
                mask[start_x: end_x, start_y: end_y] *= mm

            mask = (mask - 1) * -1
            map_env[:, :, 2] *= mask
            return map_env


    # not needed !!!!!!
    def build_collision_map(self):
        self.map_coll[:, :, 0] = np.rint(self.map[:, :, 0] - 0.35) * self.map[:, :, 0]  # 1 - Wall, 1 - Rock, all other - 0
        # base team 1
        # self.map_coll[self.PIX_CELL:self.PIX_CELL*(self.width-2), self.PIX_CELL:self.PIX_CELL*2, 0] = \
            # self.map[self.PIX_CELL:self.PIX_CELL*(self.width-2), self.PIX_CELL:self.PIX_CELL*2, 1]
        # base team 2
        # self.map_coll[self.PIX_CELL:self.PIX_CELL * (self.width - 2), self.PIX_CELL*(self.height-2):self.PIX_CELL * (self.height-1), 0] = \
        #     self.map[self.PIX_CELL:self.PIX_CELL * (self.width - 2), self.PIX_CELL*(self.height-2):self.PIX_CELL * (self.height-1), 2]


    def map_generate_random(self):
        # each layer of map mean:
        # 0 - obstacles {'land': 0, 'bush': 0.14, 'desert': 0.29, 'forest': 0.43, 'water': 0.57, 'swamp': 0.71, 'wall': 0.86, 'rock': 1}
        # 1 - red team (from 0.1 - 1 type of tanks: simple, freezer, artillery, laser, miner, repairer, heavy)
        # 2 - blue team with same types
        # 3 - Bullets
        # LAST -  fog of war (not sending)
        M = self.PIX_CELL
        self.map_env = np.zeros((self.width, self.height, 4))  # map for input AI players  Layers: 0 - obstacles, 1 - friend team, 2 - enemy`s team, 3 - bullets (because of rockets
        self.map_coll = np.zeros((self.width*M, self.height*M, 3))  # layers: 0-obstacles; 1-moving objects. 2- bullets| All with id numbers. Obstacles ids from 1. Tanks ids from 101++


        # Adding obstacles randomly on map. 2 lines from team sides is free (land)
        for y in np.arange(3, self.height - 3, 1):
            for x in np.arange(1, self.width-1, 1):
                # 50% land 50% random obstacle
                if random.randint(0, 9) > 4:
                    obstacle = self.map_obs_d[random.choice(self.map_obs)]
                else:
                    obstacle = 0  # Land
                # self.map[x*M:(x+1)*M, y*M:(y+1)*M, 0] = obstacle
                self.map_env[x, y, 0] = obstacle
                if obstacle > 0.85:
                    self.map_coll[x*M:(x+1)*M, y*M:(y+1)*M, 0] = obstacle
        # Border of map
        self.map_coll[:, 0:M, 0] = 1
        self.map_coll[ :, (self.height-1) * M:(self.height)* M,0] = 1
        self.map_coll[0:M, :, 0] = 1
        self.map_coll[(self.width-1)*M:self.width*M, :, 0] = 1
        self.map_env[:, [0, -1], 0] = 1
        self.map_env[[0, -1], :, 0] = 1

        # base for team (occupy for win) 2 cells
        base_place = int((self.width-1) / 2)
        # self.map[base_place*M:(base_place+2)*M, 1*M: 2*M, 1] = 1
        # self.map[base_place*M:(base_place+2)*M, (self.height-2)*M:(self.height-1)*M, 2] = 1
        self.map_env[base_place:(base_place+2), 1, 1] = 1
        self.map_env[base_place:(base_place+2), (self.height-2), 2] = 1
        # base coordinates: ["team", "array of Xs or Ys", "start coord, end coord"] in Pixels!
        self.map_base_xy = np.array([[[base_place*M, (base_place+2)*M], [0, M]],
                                     [[base_place*M, (base_place+2)*M], [(self.height-2)*M, (self.height-1)*M]]])

        # self.build_collision_map()  # dont need more

        # Adding tanks
        # TODO Now only simple tank types. Change to different.
        #  Need to Change it with creation of tank objects
        free_cells = set(np.arange(1, self.width-1))
        free_cells.remove(base_place)
        free_cells.remove(base_place+1)
        team_free_cells = [copy(free_cells), copy(free_cells)]
        del(free_cells)


        #  putting tanks from team 1 to map, layer 1
        y_pos = 1
        for i in range(len(self.team1)):
            tank_place = random.choice(list(team_free_cells[0]))
            self.team1[i].Y = y_pos
            self.team1[i].X = tank_place
            self.team1[i].calc_tank_coordinates(tank_place*M, y_pos*M)
            self.team1[i].player.start_side = 'up'

            # self.map[self.team1[i].coords_xy[0], self.team1[i].coords_xy[1], 1] = self.tank_type_d[self.team1[i].type]
            self.map_env[tank_place: tank_place+max(int(self.team1[i].width), 1),
                            y_pos:y_pos+ max(1, int(self.team1[i].height)), 1] = self.tank_type_d[self.team1[i].type]
            self.map_coll[self.team1[i].coords_xy[0], self.team1[i].coords_xy[1], 1] = self.team1[i].id_game
            team_free_cells[0].remove(tank_place)

        #  team 2 to map, layer 2
        y_pos = self.height-2
        for i in range(len(self.team2)):
            tank_place = random.choice(list(team_free_cells[1]))
            self.team2[i].direction_tank = 0.5
            self.team2[i].Y = y_pos
            self.team2[i].X = tank_place
            self.team2[i].calc_tank_coordinates(tank_place * M, y_pos*M)
            self.team2[i].player.start_side = 'down'

            # self.map[self.team2[i].coords_xy[0], self.team2[i].coords_xy[1], 2] = self.tank_type_d[self.team2[i].type]
            self.map_env[tank_place: tank_place+max(int(self.team2[i].width), 1),
                            y_pos:y_pos + max(1, int(self.team2[i].height)), 2] = self.tank_type_d[self.team2[i].type]
            self.map_coll[self.team2[i].coords_xy[0], self.team2[i].coords_xy[1], 1] = self.team2[i].id_game
            team_free_cells[1].remove(tank_place)

        del(team_free_cells)

    # {id, step, [reward, comment]}
    def reward(self, id_tank, score, comment):
        if id_tank >= self.ID_START:
            id_tank -= self.ID_START
        # if id_tank == 0:
            # print('step:', self.steps, score, '|', comment)
        self.rewards[id_tank, self.steps] += score * (1 - TEAM_SPIRIT)
        if id_tank in self.team1_ids:
            self.rewards[self.team1_ids, self.steps] += score * TEAM_SPIRIT
        else:
            self.rewards[self.team2_ids, self.steps] += score * TEAM_SPIRIT
        self.rewards_comment[id_tank, self.steps] += (comment + ',')

    # export rewards history
    # in: real id tank, file to save
    def rewards_to_csv(self, id_tank, file_name):

        df = pd.DataFrame()
        df['score'] = self.rewards[id_tank-self.ID_START, :self.steps+1]
        df['comment'] = self.rewards_comment[id_tank - self.ID_START, :self.steps+1]

        folder = './/player//players data//' + self.id_tanks[id_tank].name + '//'
        if not path.exists(folder):
            mkdir(path.dirname(folder))
        file_name = folder+file_name+'.csv'
        df.to_csv(file_name, index=None)
        del(df)


    def map_generate_shooting(self, num_players):

        M = self.PIX_CELL
        self.map_env = np.zeros((self.width, self.height,
                                 4))  # map for input AI players  Layers: 0 - obstacles, 1 - friend team, 2 - enemy`s team, 3 - bullets (because of rockets
        self.map_coll = np.zeros((self.width * M, self.height * M,
                                  3))  # layers: 0-obstacles; 1-moving objects. 2- bullets| All with id numbers. Obstacles ids from 1. Tanks ids from 101++

        # Adding obstacles randomly on map.
        for i in range(int(self.width*self.height*0.15)):
            obs_x = random.randint(1, self.width-1)
            obs_y = random.randint(1, self.height-1)
            obstacle = self.map_obs_d[random.choice(self.map_obs)]
            if self.map_env[obs_x, obs_y, 0] == 0:
                self.map_env[obs_x, obs_y, 0] = obstacle
                if obstacle > 0.85:
                    self.map_coll[obs_x * M:(obs_x + 1) * M, obs_y * M:(obs_y + 1) * M, 0] = obstacle

        # Border of map
        self.map_coll[:, 0:M, 0] = 1
        self.map_coll[:, (self.height - 1) * M:(self.height) * M, 0] = 1
        self.map_coll[0:M, :, 0] = 1
        self.map_coll[(self.width - 1) * M:self.width * M, :, 0] = 1
        self.map_env[:, [0, -1], 0] = 1
        self.map_env[[0, -1], :, 0] = 1

        # base coordinates: ["team", "array of Xs or Ys", "start coord, end coord"] in Pixels!
        self.map_base_xy = np.array([[[0, 0], [0, 0]],
                                     [[0, 0],
                                      [0, 0]]])

        # Adding tanks
        #  putting tanks from team 1 to map, layer 1
        y_pos = self.height // 2
        half_x = num_players // 2
        x_pos = np.arange(self.width // 2 - half_x, self.width // 2 - half_x + num_players, 1)

        # dummy possitions
        x_cent = self.width // 2
        y_cent = self.height // 2
        x_rand = random.randint(-1, 1)
        y_rand = random.randint(-1, 1)
        x_dummy = np.array([x_cent, self.width-4, x_cent, 3]) + x_rand
        y_dummy = np.array([3, y_cent, self.height-4, y_cent]) + y_rand
        dir_dummy = np.array([0, 0.75, 0.5, 0.25]) + (np.random.random((4)) - 0.5)
        i_p = 0
        i_d = 0
        for i in range(len(self.team1)):
            if self.team1[i].name == 'dummy':
                self.team1[i].Y = y_dummy[i_d]
                self.team1[i].X = x_dummy[i_d]
                self.team1[i].direction_tank = dir_dummy[i_d]
                i_d += 1
            else:
                self.team1[i].Y = y_pos
                self.team1[i].X = x_pos[i_p]
                i_p += 1
            self.team1[i].calc_tank_coordinates(self.team1[i].X * M, self.team1[i].Y * M)
            self.team1[i].player.start_side = 'up'

            self.map_env[self.team1[i].X: self.team1[i].X + max(int(self.team1[i].width), 1),
            self.team1[i].Y:self.team1[i].Y + max(1, int(self.team1[i].height)), 1] = self.tank_type_d[self.team1[i].type]
            self.map_coll[self.team1[i].coords_xy[0], self.team1[i].coords_xy[1], 1] = self.team1[i].id_game
            self.map_env[self.team1[i].X, self.team1[i].Y, 0] = 0  # under tank - plain


        # x_dummy = np.array([x_cent-2, x_cent+2, self.width-4, self.width-4, x_cent-2, x_cent+2, 3, 3]) + x_rand
        # y_dummy = np.array([3, 3, y_cent-2, y_cent+2, self.height-4, self.height-4, y_cent-2, y_cent+2]) + y_rand
        x_dummy = np.array([self.width - 4, self.width - 4, x_cent - 2, x_cent + 2, 3, 3]) + x_rand
        y_dummy = np.array([y_cent - 2, y_cent + 2, self.height - 4, self.height - 4, y_cent - 2, y_cent + 2]) + y_rand
        dir_dummy = [0, 0.75, 0.5, 0.25] + (np.random.random((4)) - 0.5)
        #  team 2 to map, layer 2
        for i in range(len(self.team2)):
            self.team2[i].direction_tank = dir_dummy[i//2]
            self.team2[i].Y = y_dummy[i]
            self.team2[i].X = x_dummy[i]
            self.team2[i].calc_tank_coordinates(self.team2[i].X * M, self.team2[i].Y * M)
            self.team2[i].player.start_side = 'down'

            self.map_env[self.team2[i].X: self.team2[i].X + max(int(self.team2[i].width), 1),
            self.team2[i].Y:self.team2[i].Y + max(1, int(self.team2[i].height)), 2] = self.tank_type_d[self.team2[i].type]
            self.map_coll[self.team2[i].coords_xy[0], self.team2[i].coords_xy[1], 1] = self.team2[i].id_game
            self.map_env[self.team2[i].X, self.team2[i].Y, 0] = 0  # under tank - plain






