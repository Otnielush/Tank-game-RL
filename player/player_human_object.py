import keyboard


class player_human():
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.score = 0
        self.last_action = 0

    def __str__(self):
        return str(self.name)+' id: '+str(self.id)

    def take_action(self):
        if keyboard.is_pressed("w"):
            print("You pressed w")
            self.last_action = 'forward'

        if keyboard.is_pressed("s"):
            print("You pressed s")
            self.last_action = 'back'

        if keyboard.is_pressed("a"):
            print("You pressed a")
            self.last_action = 'left'

        if keyboard.is_pressed("d"):
            print("You pressed d")
            self.last_action = 'right'

        return self.last_action





