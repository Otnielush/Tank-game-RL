

# Steps: create object, change_tank_type, new game -> connected_new_game
class player_AI():
    def __init__(self, name, difficulty=1):
        self.id_connection = 0
        self.id_game = 0
        self.name = name
        self.connection = 0
        self.env = 0
        self.tank_type = ''

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
        self.env = self.connection.get_env_from_server(self.id)
        return self.action_function()

    # TODO func for save env and rewards



