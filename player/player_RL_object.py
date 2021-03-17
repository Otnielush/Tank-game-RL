from numpy import argmax



# Main RL player
# input neural network model
# TODO: take parent from AI and write action function for RL
class player_RL():
    def __init__(self, id, model ):
        self.id = id
        self.model = model

    def __str__(self):
        return self.id

    # input data/info/environment/status of game
    # output action to make move
    def take_action(self, environment):
        return argmax(self.model.predict(environment))


