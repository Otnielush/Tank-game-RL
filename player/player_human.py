import keyboard

if __name__ == '__main__': # for test
    class player_obj():
        def __init__(self, name):
            pass
else:
    from .player_superclass import player_obj

class player_human(player_obj):
    def __init__(self, name):
        super(player_human, self).__init__(name)
        self.id = id
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

    def move(self):
        pass





