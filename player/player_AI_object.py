

class player_AI():
    def __init__(self, id, difficulty=1):
        self.id = id
        if difficulty == 1:
            self.strategy = "function for moving"
        elif difficulty == 2:
            self.strategy = "another function for moving"

    def __str__(self):
        return self.id

