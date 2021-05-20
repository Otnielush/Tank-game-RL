from os import path, mkdir
import json


# Steps: create object, change_tank_type, new game -> connected_new_game
class player_obj():
    def __init__(self, name, difficulty=1):
        self.id_connection = 0
        self.id_game = 0
        self.name = name
        self.connection = 0
        self.env = 0
        self.data = 0
        self.reward = 0
        self.info = 0
        self.tank_type = ''
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

        if difficulty == 1:
            self.action_function = lambda x: 1, 0.1, -0.1, False, False   # TODO: write action function for AI
        elif difficulty == 2:
            self.action_function = "another function for moving"

    def __str__(self):
        return "'{}' id_conn: {}, tank: {}, games: {}, win rate {}%, destroyed: {}".format(self.name, self.id_connection,
                            self.tank_type, self.games_played, round(self.wins/self.games_played*100), self.tanks_killed)

    # when connected to Waitroom receive id
    def change_id(self, id_conn=-1, id_game=-1):
        if id_conn != -1:
            self.id_connection = id_conn
        if id_game != -1:
            self.id_game = id_game

    # ID will receive earlier, but now its so
    def connected_new_game(self, net_connection):
        self.connection = net_connection

    def change_tank_type(self, type):
        self.tank_type = type

    def move(self):
        self.env, self.data, self.reward, self.info = self.connection.get_env_from_server(self.id_game)
        action = self.action_function(self)

        # TODO stopped here
        self.connection.send_action(self.id_game, action)
        # TODO change id for tanks. player dont know ID_START - write it in broadcasting

    # game round done
    def done(self):
        self.tank_ingame.accuracy = round(self.tank_ingame.num_hits / (self.tank_ingame.num_shots+0.01) * 100, 1)
        self.num_shots += self.tank_ingame.num_shots
        self.num_hits += self.tank_ingame.num_hits


    def load_player(self):
        file_config = './/player//players data//'+self.name+'//config.json'
        if path.exists(file_config):
            with open(file_config, 'r') as cfg:
                config = json.load(cfg)

            params2load = ['games_played', 'wins', 'tanks_killed', 'deaths', 'bases_captured', 'draws', 'num_shots', 'num_hits']
            for key in params2load:
                try:
                    self.__dict__[key] = config[key]
                except:
                    self.__dict__[key] = 0


    def save_model(self):
        folder = './/player//players data//' + self.name + '//'
        if not path.exists(folder):
            mkdir(path.dirname(folder))
        data = dict()
        params2load = ['games_played', 'wins', 'tanks_killed', 'deaths', 'bases_captured', 'draws', 'num_shots', 'num_hits']
        for key in params2load:
            data[key] = self.__dict__[key]

        with open(folder + '//config.json', 'w') as cfg:
            json.dump(data, cfg)

        hist_data = [str(self.__dict__[k]) for k in params2load]
        with open(folder + '//history.csv', 'a') as the_file:
            the_file.write(','.join(hist_data)+'\n')


