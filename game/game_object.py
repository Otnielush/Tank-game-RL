import numpy as np
import random
from copy import copy
from tank.tank_object import tank_type, Tank
from net.broadcasting import net_connection

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
        self.connection = 0


        # SCORES
        self.score_win          = 5
        self.score_hit          = 1
        self.score_kill         = 2
        self.score_dmg          = 0.01
        self.score_death        = -2
        self.score_kill_assist  = 1
        self.score_exploring    = 1
        self.friendly_fire      = -1


    # start new round with array of connected players [player, ...]
    # id starts from 101
    def new_game(self, height, width, team1_payers, team2_payers):
        self.width = width
        self.height = height

        self.team1 = []  # [Tank]
        self.team2 = []
        self.ID_START = 101
        self.id_tanks = self.ID_START
        self.bullets = []
        self.id_b = 200   # id bullets
        num_players = len(team1_payers) + len(team2_payers)
        self.connection = net_connection(num_players, False, (height, width, 4), 9, (5,))

        # Creating object Tank for players and sending connection
        for i in range(len(team1_payers)):
            self.team1.append(Tank(self.id_tanks, team1_payers[i], 0, 0, self.PIX_CELL))
            team1_payers[i].change_id(id_game=self.id_tanks)
            self.team1[i].player.connected_new_game(self.connection)
            self.id_tanks += 1
        for i in range(len(team2_payers)):
            self.team2.append(Tank(self.id_tanks, team2_payers[i], 0, 0, self.PIX_CELL))
            team2_payers[i].change_id(id_game=self.id_tanks)
            self.team2[i].player.connected_new_game(self.connection)
            self.id_tanks += 1


        self.map_generate()

        # sending ENV to network connection
        env_team1 = self.build_env_map_team(1)
        for i in range(len(self.team1)):
            self.connection.send_env_to_players(self.team1[i].id_game - self.ID_START, env_team1,
                    [self.team1[i].Y, self.team1[i].X, self.team1[i].direction_tank, self.team1[i].direction_tower, self.team1[i].hp,
                     self.team1[i].speed, self.team1[i].reloading_ammo, self.team1[i].reloading_skill, self.team1[i].ammunition])
        env_team2 = self.build_env_map_team(2)
        for i in range(len(self.team2)):
            self.connection.send_env_to_players(self.team2[i].id_game - self.ID_START, env_team2,
                    [self.team2[i].Y, self.team2[i].X, self.team2[i].direction_tank, self.team2[i].direction_tower, self.team2[i].hp,
                     self.team2[i].speed, self.team2[i].reloading_ammo, self.team2[i].reloading_skill, self.team2[i].ammunition])


    # Making input for players
    def build_env_map_team(self, team_num):
        if team_num == 1:
            map_env = copy(self.map_env[:, :, :4])
            mask = np.ones(map_env.shape[:2])
            for i in range(len(self.team1)):
                start_x = max(self.team1[i].X - self.team1[i].sight_range, 0)
                end_x = min(self.team1[i].X + self.team1[i].sight_range +1, self.width)
                start_y = max(self.team1[i].Y - self.team1[i].sight_range, 0)
                end_y = min(self.team1[i].Y + self.team1[i].sight_range +1, self.height)
                start_y_m = max(self.team1[i].sight_range - self.team1[i].Y, 0)
                start_x_m = max(self.team1[i].sight_range - self.team1[i].X, 0)
                mm = self.team1[i].sight_mask[start_y_m: start_y_m + min(self.height - start_y, self.team1[i].sight_range * 2 + 1 - start_y_m),
                                                            start_x_m: start_x_m + min(self.width - start_x, self.team1[i].sight_range * 2 + 1 - start_x_m)]
                # print(mm)
                mask[start_y: end_y , start_x: end_x] *= mm

            mask = (mask - 1) * -1
            map_env[:, :, 2] *= mask

            return map_env

        else:
            map_env = copy(self.map_env[:, :, [0, 2, 1, 3]])   # 1- friendly team, 2 - enemy`s team
            mask = np.ones(map_env.shape[:2])
            for i in range(len(self.team2)):
                start_x = max(self.team2[i].X - self.team2[i].sight_range, 0)
                end_x = min(self.team2[i].X + self.team2[i].sight_range + 1, self.width)
                start_y = max(self.team2[i].Y - self.team2[i].sight_range, 0)
                end_y = min(self.team2[i].Y + self.team2[i].sight_range + 1, self.height)
                start_y_m = max(self.team2[i].sight_range - self.team2[i].Y, 0)
                start_x_m = max(self.team2[i].sight_range - self.team2[i].X, 0)
                mm = self.team2[i].sight_mask[
                     start_y_m: start_y_m + min(self.height - start_y, self.team2[i].sight_range * 2 + 1 - start_y_m),
                     start_x_m: start_x_m + min(self.width - start_x, self.team2[i].sight_range * 2 + 1 - start_x_m)]
                mask[start_y: end_y, start_x: end_x] *= mm

            mask = (mask - 1) * -1
            map_env[:, :, 2] *= mask

            return map_env



    def build_collision_map(self):
        self.map_coll[:,:,0] = np.rint(self.map[:,:,0] - 0.35)  # 1 - Wall, 1 - Rock, all other - 0
        self.map_coll[0:self.PIX_CELL, :, 0] = self.map[0:self.PIX_CELL, :, 1]  # base team 1
        self.map_coll[(self.height-1)*self.PIX_CELL:, :, 0] = self.map[(self.height - 1) * self.PIX_CELL:, :, 2]  # base team 2

    def map_generate(self):
        # each layer of map mean:
        # 0 - obstacles {'land': 0, 'bush': 0.14, 'desert': 0.29, 'forest': 0.43, 'water': 0.57, 'swamp': 0.71, 'wall': 0.86, 'rock': 1}
        # 1 - red team (from 0.1 - 1 type of tanks: simple, freezer, artillery, laser, miner, repairer, heavy)
        # 2 - blue team with same types
        # 3 - Bullets
        # LAST -  fog of war (not sending)
        M = self.PIX_CELL
        self.map = np.zeros((self.height*M, self.width*M, 5))  # map for game/video
        self.map_env = np.zeros((self.height, self.width, 4))  # map for input AI players  Layers: 0 - obstacles, 1 - friend team, 2 - enemy`s team, 3 - bullets (because of rockets
        self.map_coll = np.zeros((self.height*M, self.width*M, 3))  # layers: 0-obstacles; 1-moving objects. 2- bullets| All with id numbers. Obstacles ids from 1. Tanks ids from 101++


        # Adding obstacles randomly on map. 2 lines from team sides is free (land)
        for y in np.arange(3, self.height - 3, 1):
            for x in np.arange(1, self.width-1, 1):
                obstacle = self.map_obs_d[random.choice(self.map_obs)]
                self.map[y*M:(y+1)*M, x*M:(x+1)*M, 0] = obstacle
                self.map_env[y, x, 0] = obstacle
        # Border of map
        self.map[0:M, :, 0] = 1
        self.map[(self.height-1) * M:(self.height)* M, :, 0] = 1
        self.map[:, 0:M, 0] = 1
        self.map[:, (self.width-1)*M:self.width*M, 0] = 1
        self.map_env[[0, -1], :, 0] = 1
        self.map_env[:, [0, -1], 0] = 1

        # base for team (occupy for win) 2 cells
        base_place = int((self.width-1) / 2)
        self.map[1*M: 2*M, base_place*M:(base_place+2)*M, 1] = 1
        self.map[(self.height-2)*M:, base_place*M:(base_place+2)*M, 2] = 1
        self.map_env[1, base_place:(base_place+2), 1] = 1
        self.map_env[(self.height-2), base_place:(base_place+2), 2] = 1

        self.build_collision_map()

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
            tank_place = 10 # !!!!! TEST
            self.team1[i].Y = y_pos
            self.team1[i].X = tank_place
            self.team1[i].calc_tank_coordinates(y_pos*M, tank_place*M)

            self.map[self.team1[i].coords_yx[0], self.team1[i].coords_yx[1], 1] = self.tank_type_d[self.team1[i].type]
            self.map_env[y_pos:y_pos+ max(1, round(self.team2[i].height)),
                        tank_place: tank_place+max(round(self.team1[i].width), 1), 1] = self.tank_type_d[self.team1[i].type]
            self.map_coll[self.team1[i].coords_yx[0], self.team1[i].coords_yx[1], 1] = self.team1[i].id_game
            team_free_cells[0].remove(tank_place)

        #  team 2 to map, layer 2
        y_pos = self.height-2
        for i in range(len(self.team2)):
            tank_place = random.choice(list(team_free_cells[1]))
            self.team2[i].direction_tank = 0.5
            self.team2[i].Y = y_pos
            self.team2[i].X = tank_place
            self.team2[i].calc_tank_coordinates(y_pos*M, tank_place * M)

            self.map[self.team2[i].coords_yx[0], self.team2[i].coords_yx[1], 2] = self.tank_type_d[self.team2[i].type]
            self.map_env[y_pos:y_pos+ max(1, round(self.team2[i].height)),
                        tank_place: tank_place+max(round(self.team2[i].width), 1), 2] = self.tank_type_d[self.team2[i].type]
            self.map_coll[self.team2[i].coords_yx[0], self.team2[i].coords_yx[1], 1] = self.team2[i].id_game
            team_free_cells[1].remove(tank_place)

        del(team_free_cells)










