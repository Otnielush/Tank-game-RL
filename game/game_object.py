import numpy as np
from collections import deque
import random
from copy import copy
from tank.tank_object import tank_type, Tank
from net.broadcasting import net_connection

# game object
# Steps: create, new_game
class TankGame():

    def __init__(self):
        self.PIXELS_IN_CELL = 5
        self.team_size = 0
        self.width = 0
        self.height = 0
        self.map_obs = ['land', 'bush', 'desert', 'forest', 'water', 'swamp', 'wall', 'rock']
        self.map_obs_d = {'land': 0, 'bush': 0.14, 'desert': 0.29, 'forest': 0.43, 'water': 0.57, 'swamp': 0.71, 'wall': 0.86, 'rock': 1}  # Dictionary for obstacles
        self.tank_type_d = {t[0]:t[1] for t in zip(tank_type, np.linspace(0, 1, len(tank_type)))}
        self.team1 = [] # [Tank]
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

        self.team1 = [] # [Tank]
        self.team2 = []
        ID_START = 101
        self.id_tanks = ID_START
        num_players = len(team1_payers) + len(team2_payers)
        self.connection = net_connection(num_players, False, (height, width, 4), 9, (5,))

        # Creating object Tank for players and sending connection
        for i in range(len(team1_payers)):
            self.team1.append(Tank(self.id_tanks, team1_payers[i], 0, 0))
            team1_payers[i].change_id(id_game=self.id_tanks)
            self.team1[i].player.connected_new_game(self.connection)
            self.id_tanks += 1
        for i in range(len(team2_payers)):
            self.team2.append(Tank(self.id_tanks, team2_payers[i], 0, 0))
            team2_payers[i].change_id(id_game=self.id_tanks)
            self.team2[i].player.connected_new_game(self.connection)
            self.id_tanks += 1


        self.map_generate()
        self.build_collision_map()

        # sending ENV to network connection
        env_team1 = self.build_env_map_team(1)
        for i in range(len(self.team1)):
            self.connection.send_env_to_players(self.team1[i].id_game - ID_START, env_team1,
                    [self.team1[i].Y, self.team1[i].X, self.team1[i].direction_tank, self.team1[i].direction_tower, self.team1[i].hp,
                     self.team1[i].speed, self.team1[i].reloading_ammo, self.team1[i].reloading_skill, self.team1[i].ammunition])
        env_team2 = self.build_env_map_team(2)
        for i in range(len(self.team2)):
            self.connection.send_env_to_players(self.team2[i].id_game - ID_START, env_team2,
                    [self.team2[i].Y, self.team2[i].X, self.team2[i].direction_tank, self.team2[i].direction_tower, self.team2[i].hp,
                     self.team2[i].speed, self.team2[i].reloading_ammo, self.team2[i].reloading_skill, self.team2[i].ammunition])




    def build_collision_map(self):
        self.collision_map = np.rint(self.map[:,:,0] - 0.35) # 0.35 keeping only colliding obstacles

    def map_generate(self):
        # each layer of map mean:
        # 0 - obstacles {'land': 0, 'bush': 0.14, 'desert': 0.29, 'forest': 0.43, 'water': 0.57, 'swamp': 0.71, 'wall': 0.86, 'rock': 1}
        # 1 - red team (from 0.1 - 1 type of tanks: simple, freezer, artillery, laser, miner, repairer, heavy)
        # 2 - blue team with same types
        # 3 - Bullets
        # LAST -  fog of war (not sending)
        M = self.PIXELS_IN_CELL
        self.map = np.zeros((self.height*M, self.width*M, 5))  # map for game/collision/video
        self.map_env = np.zeros((self.height, self.width, 5))  # map for input AI players


        # Adding obstacles randomly on map. 2 lines from team sides is free (land)
        for y in np.arange(2, self.height - 2, 1):
            for x in range(self.width):
                obstacle = self.map_obs_d[random.choice(self.map_obs)]
                self.map[y*M:(y+1)*M, x*M:(x+1)*M, 0] = obstacle
                self.map_env[y, x, 0] = obstacle

        # base for team (occupy for win) 2 cells
        base_place = int((self.width-1) / 2)
        self.map[0:M, base_place*M:(base_place+2)*M, 1] = 1
        self.map[(self.height-1)*M:, base_place*M:(base_place+2)*M, 2] = 1
        self.map_env[0:2, base_place:(base_place+2), 1] = 1
        self.map_env[(self.height-1):, base_place:(base_place+2), 2] = 1

        # Adding tanks
        # TODO Now only simple tank types. Change to different.
        #  Need to Change it with creation of tank objects
        free_cells = set(np.arange(self.width))
        free_cells.remove(base_place)
        free_cells.remove(base_place+1)
        team_free_cells = [copy(free_cells), copy(free_cells)]
        del(free_cells)

        # [0, 1] 0 - y place on map, 1 - layer on map for team
        # TODO: rebuild
        team = [[0, 1], [self.height-1, 2]]
        for i in range(2):
            for j in range(self.team_size):
                tank_place = random.choice(list(team_free_cells[i]))
                self.map[team[i][0]*M:(team[i][0]+1)*M, tank_place*M:(tank_place+1)*M, team[i][1]] = self.tank_type_d['simple']
                team_free_cells[i].remove(tank_place)













# NOT NEEDED
    def reset(self, width=0, height=0, team_size=0):
        if team_size != 0:
            self.team_size = team_size
        if width == 0 or height == 0:
            width = self.width
            height = self.height

        self.map_generate()

        return self.map




# save all actions and data into game object or may be each player will save it himself
class ReplayBuffer():

    def __init__(self, max_size):
        self.max_size = max_size
        self.transitions = deque()

    def add(self, observation, action, reward, observation2):
        if len(self.transitions) > self.max_size:
            self.transitions.popleft()
        self.transitions.append((observation, action, reward, observation2))

    def sample(self, count):
        # samples = deque(list(self.transitions)[len(self.transitions)-count:])
        # self.transitions = deque(list(self.transitions)[:len(self.transitions)-count], self.max_size)
        # return random.sample(samples, count)

        # return random.sample(self.transitions, count)

        # taking a random part of buffer
        buffer_start = random.randint(0, len(self.transitions) - count)
        return deque(itertools.islice(self.transitions, buffer_start, buffer_start + count))

    def size(self):
        return len(self.transitions)



