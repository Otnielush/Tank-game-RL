from .player_superclass import player_obj



class player_AI(player_obj):
    def __init__(self, name, difficulty=1):
        super(player_AI, self).__init__(name, difficulty)

    def move(self):
        pass