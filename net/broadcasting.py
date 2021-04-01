# Emulation of net broadcastiong
# Simple saving sended/received to variables

import numpy as np



# env_size = (x, y, z,...)
# act_size = (x, y, z,...)
class net_connection():
    def __init__(self, num_players, deleting_last_action, env_size, data_size, act_size):
        self.num_players = num_players
        self.env_size = env_size
        self.act_size = act_size
        self.env_from_server = np.zeros((num_players,) + env_size)
        self.data_from_server = np.zeros((num_players,) + (data_size,))  #10: reward, x, y, angle_tank, angle_tower, hp, speed, (time to reload: ammo, skill); ammunition
        self.data_from_players = np.zeros((num_players,) + act_size)

        # after getting action need to zeroing?
        # if disconnected - stop
        # backwards - repeating last action
        self.deleting_last_action = deleting_last_action

    # input id player, (width, height, num layers)
    # data: hp, speed, time to reload: ammo, skill; ammunition
    def send_env_to_players(self, id, env, data):
        self.env_from_server[id] = env
        self.data_from_server[id] = data

    # player asking for envinronment
    def get_env_from_server(self, id):
        return self.env_from_server[id], self.data_from_server[id]

    # id of player, action
    def send_action(self, id, action):
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
