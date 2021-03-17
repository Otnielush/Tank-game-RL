# Emulation of net broadcastiong
# Simple saving sended/received to variables

import numpy as np


# env_size = (x, y, z,...)
# act_size = (x, y, z,...)
class connection():
    def __init__(self, team_size, deleting_last_action, env_size, act_size):
        self.team_size = team_size
        self.env_size = env_size
        self.act_size = act_size
        self.data_from_server = np.zeros((team_size*2,) + env_size)
        self.data_from_players = np.zeros((team_size*2,) + act_size)
        # after getting action need to zeroing?
        # if disconnected - stop
        # backwards - repeating last action
        self.deleting_last_action = deleting_last_action

    # input id player, (width, height, num layers)
    def send_env_to_players(self, id, data):
        self.data_from_server[id] = data

    # player asking for envinronment
    def get_env_from_server(self, id):
        return self.data_from_server[id]

    # id of player, action
    def send_action(self, id, action):
        self.data_from_players[id] = action

    # server receive actions from players
    def get_actions(self):
        data = self.data_from_players
        if self.deleting_last_action:
            # self.data_from_server = np.zeros((self.team_size*2,) + self.env_size)
            self.data_from_players = np.zeros((self.team_size*2,) + self.act_size)

        return data

    def __str__(self):
        return 'team size: {}, deleting last action? {}\nenv size: {}\naction size: {}'.format(
            self.team_size, self.deleting_last_action, self.env_size, self.act_size
        )

