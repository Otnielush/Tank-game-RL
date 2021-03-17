from net.broadcasting import net_connection


class player_AI(net_connection):
    def __init__(self, name, difficulty=1):
        self.id = 0
        self.name = name
        self.connection = 0
        self.env = 0
        self.tank_type = ''
        print("hi")
        print(("Hello"))

        if difficulty == 1:
            self.action_function = lambda x: 1, 0.1, -0.1, False, False   # TODO: write action function for AI
        elif difficulty == 2:
            self.action_function = "another function for moving"

    def __str__(self):
        return self.id


    # ID will receive earlier, but now its so
    def connected_new_game(self, net_connection, id):
        self.id = id
        self.connection = net_connection

    def change_tank_type(self, type):
        self.tank_type = type

    def move(self):
        self.env = self.connection.get_env_from_server(self.id)
        return self.action_function()




