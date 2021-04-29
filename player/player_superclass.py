


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

        if difficulty == 1:
            self.action_function = lambda x: 1, 0.1, -0.1, False, False   # TODO: write action function for AI
        elif difficulty == 2:
            self.action_function = "another function for moving"

    def __str__(self):
        return "'{}' id_conn: {}, tank: {}".format(self.name, self.id_connection, self.tank_type)

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
        pass
