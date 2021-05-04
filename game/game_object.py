import numpy as np
import random
from copy import copy
from tank.tank_object import tank_type, Tank
from net.broadcasting import net_connection
import time

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
        self.rewards = 0  # {id, [step, reward, comment]}  # new game generates
        self.steps = 0
        self.time_round_len = 5*60  # seconds


        # SCORES
        self.score_win              = 5
        self.score_lose             = -5
        self.score_hit              = 1
        self.score_kill             = 2
        self.score_dmg              = 0.01
        self.score_death            = -2
        self.score_kill_assist      = 1
        self.score_exploring        = 1
        self.score_friendly_fire    = -1
        self.score_move             = -0.01
        self.score_shot             = -0.01
        self.score_take_hit         = self.score_hit * -0.5
        self.score_take_dmg         = self.score_dmg * -1
        self.score_capture          = 0.01


        self.data = 0
        self.frame_step = 0


    # start new round with array of connected players [player, ...]
    # id starts from 101
    def new_game(self, width, height, team1_payers, team2_payers, VIDEO):
        self.width = width
        self.height = height
        self.VIDEO = VIDEO

        self.steps = 0
        self.team1 = []  # [Tank]
        self.team2 = []
        self.ID_START = 101
        id_tank = self.ID_START
        self.bullets = []
        self.bullets_in_act = []  # ids of bullets - id_bul ( for bullets array )
        self.id_bul = 200   # id bullets
        num_players = len(team1_payers) + len(team2_payers)
        self.connection = net_connection(num_players, False, (width, height, 4), 10, (5,))

        # Creating object Tank for players and sending connection
        for i in range(len(team1_payers)):
            self.team1.append(Tank(id_tank, team1_payers[i], 1, 0, 0, self.PIX_CELL))
            team1_payers[i].change_id(id_game=id_tank)
            team1_payers[i].start_side = 'up'
            team1_payers[i].tank_ingame = self.team1[-1]
            team1_payers[i].games_played += 1
            self.team1[i].player.connected_new_game(self.connection)
            self.id_tanks[id_tank] = self.team1[-1]
            id_tank += 1
        for i in range(len(team2_payers)):
            self.team2.append(Tank(id_tank, team2_payers[i], 2, 0, 0, self.PIX_CELL))
            team2_payers[i].change_id(id_game=id_tank)
            team2_payers[i].start_side = 'down'
            team2_payers[i].tank_ingame = self.team2[-1]
            team2_payers[i].games_played += 1
            self.team2[i].player.connected_new_game(self.connection)
            self.id_tanks[id_tank] = self.team2[-1]
            id_tank += 1

        # (num_players, 20000, 2))  # {id, step, [reward, comment]}
        self.rewards = [[[0, 'comment about reward']
                         for _ in range(20000)]
                        for _ in range(num_players)]

        self.map_generate()
        # Round timer
        self.time_start = time.perf_counter()
        # sending ENV to network connection
        self.send_data_to_players({'game_start': True})



    def send_data_to_players(self, info):
        timer = (self.time_round_len - (time.perf_counter() - self.time_start)) / self.time_round_len
        env_team1 = self.build_env_map_team(1)
        for i in range(len(self.team1)):
            idd = self.team1[i].id_game - self.ID_START
            # env_map
            # 10: x, y, angle_tank, angle_tower, hp, speed, (time to reload: ammo, skill); ammunition; round time left in %;
            # reward; info(start game, game done);
            self.connection.send_env_to_players(idd, env_team1,
                    [self.team1[i].X, self.team1[i].Y, self.team1[i].direction_tank, self.team1[i].direction_tower, self.team1[i].hp,
                     self.team1[i].speed, self.team1[i].reloading_ammo, self.team1[i].reloading_skill, self.team1[i].ammunition, timer],
                    copy(self.rewards[idd][self.steps][0]), info)
        env_team2 = self.build_env_map_team(2)
        for i in range(len(self.team2)):
            idd = self.team2[i].id_game - self.ID_START
            self.connection.send_env_to_players(idd, env_team2,
                    [self.team2[i].X, self.team2[i].Y, self.team2[i].direction_tank, self.team2[i].direction_tower, self.team2[i].hp,
                     self.team2[i].speed, self.team2[i].reloading_ammo, self.team2[i].reloading_skill, self.team2[i].ammunition, timer],
                    copy(self.rewards[idd][self.steps][0]), info)


    # Making input for players
    def build_env_map_team(self, team_num):
        if team_num == 1:
            map_env = copy(self.map_env[:, :, :4])
            mask = np.ones(map_env.shape[:2])
            for i in range(len(self.team1)):
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

                mm = self.team2[i].sight_mask[start_x_m: end_x_m, start_y_m: end_y_m]
                mask[start_x: end_x, start_y: end_y] *= mm

            mask = (mask - 1) * -1
            map_env[:, :, 2] *= mask
            return map_env

        else:
            map_env = copy(self.map_env[:, :, [0, 2, 1, 3]])   # Layers: 1- friendly team, 2 - enemy`s team
            mask = np.ones(map_env.shape[:2])
            for i in range(len(self.team2)):
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


    def map_generate(self):
        # each layer of map mean:
        # 0 - obstacles {'land': 0, 'bush': 0.14, 'desert': 0.29, 'forest': 0.43, 'water': 0.57, 'swamp': 0.71, 'wall': 0.86, 'rock': 1}
        # 1 - red team (from 0.1 - 1 type of tanks: simple, freezer, artillery, laser, miner, repairer, heavy)
        # 2 - blue team with same types
        # 3 - Bullets
        # LAST -  fog of war (not sending)
        M = self.PIX_CELL
        # self.map = np.zeros((self.width*M, self.height*M, 5))  # map for game/video
        self.map_env = np.zeros((self.width, self.height, 4))  # map for input AI players  Layers: 0 - obstacles, 1 - friend team, 2 - enemy`s team, 3 - bullets (because of rockets
        self.map_coll = np.zeros((self.width*M, self.height*M, 3))  # layers: 0-obstacles; 1-moving objects. 2- bullets| All with id numbers. Obstacles ids from 1. Tanks ids from 101++


        # Adding obstacles randomly on map. 2 lines from team sides is free (land)
        for y in np.arange(3, self.height - 3, 1):
            for x in np.arange(1, self.width-1, 1):
                obstacle = self.map_obs_d[random.choice(self.map_obs)]
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
            self.team1[i].start_side = 'up'

            # self.map[self.team1[i].coords_xy[0], self.team1[i].coords_xy[1], 1] = self.tank_type_d[self.team1[i].type]
            self.map_env[tank_place: tank_place+max(int(self.team1[i].width), 1),
                            y_pos:y_pos+ max(1, int(self.team2[i].height)), 1] = self.tank_type_d[self.team1[i].type]
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
            self.team2[i].start_side = 'down'

            # self.map[self.team2[i].coords_xy[0], self.team2[i].coords_xy[1], 2] = self.tank_type_d[self.team2[i].type]
            self.map_env[tank_place: tank_place+max(int(self.team2[i].width), 1),
                            y_pos:y_pos + max(1, int(self.team2[i].height)), 2] = self.tank_type_d[self.team2[i].type]
            self.map_coll[self.team2[i].coords_xy[0], self.team2[i].coords_xy[1], 1] = self.team2[i].id_game
            team_free_cells[1].remove(tank_place)

        del(team_free_cells)

    # {id, step, [reward, comment]}
    def reward(self, id_tank, score, comment):
        self.rewards[id_tank][self.steps][0] += score
        self.rewards[id_tank][self.steps][1] += (comment + ',')








