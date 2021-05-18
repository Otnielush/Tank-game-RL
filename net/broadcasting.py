# Emulation of net broadcastiong
# Simple saving sended/received to variables

from copy import copy, deepcopy
import numpy as np



# env_size = (x, y, z,...)
# act_size = (x, y, z,...)
class net_connection():
    def __init__(self, num_players, deleting_last_action, env_size, data_size, act_size):
        self.num_players = num_players
        self.env_size = env_size
        self.act_size = act_size
        self.env_from_server = np.zeros((num_players,) + env_size)

        # reward,
        # 10: x, y, angle_tank, angle_tower, hp, speed, (time to reload: ammo, skill);
        # ammunition; round time left in %
        self.reward_from_server = np.zeros((num_players))
        self.data_from_server = np.zeros((num_players,) + (1, data_size))
        self.info_from_server = {'game_start': False, 'game_done': False}  # game_start - starting new game; game_done - game finished

        self.data_from_players = np.zeros((num_players,) + act_size)

        # after getting action need to zeroing?
        # if disconnected - stop
        # backwards - repeating last action
        self.deleting_last_action = deleting_last_action

    # input:
    # env_map
    # 12: DATA: x, y, angle_tank, angle_tower, hp, speed, (time to reload: ammo, skill); ammunition; round time left in %;
    # reward; info(start game, game done);
    def send_env_to_players(self, id, env, data, reward, info):
        self.reward_from_server[id] = copy(reward)
        self.env_from_server[id] = deepcopy(env)
        self.data_from_server[id] = np.reshape(data, (1, -1))

        if info is not None:
            for key in info:
                self.info_from_server[key] = info[key]

    # player asking for envinronment
    def get_env_from_server(self, id):
        id = id - 101
        return self.env_from_server[id], self.data_from_server[id], self.reward_from_server[id], self.info_from_server

    # id of player, action
    def send_action(self, id, action):
        id -= 101
        self.data_from_players[id] = action

    # server receive actions from players
    def get_actions(self):
        data = self.data_from_players
        if self.deleting_last_action:
            self.data_from_players = np.zeros((self.num_players,) + self.act_size)

        return data

    def __str__(self):
        return 'number of players: {}, deleting last action? {}\nenv size: {}\naction size: {}'.format(
            self.num_players, self.deleting_last_action, self.env_size, self.act_size
        )


# Class for connected player waiting for game. NOT DONE
class waiting_room():
    def __init__(self):
        self.players = []
        self.start_id = 101

    def connect(self, name):
        id = self.start_id + len(self.players)
        self.players.append([id, name, 'simple'])

        return id

    def change_tank_type(self, id, type):
        self.players[id - 101][2] = type
