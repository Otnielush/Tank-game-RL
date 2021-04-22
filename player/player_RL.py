from numpy import argmax
# from .player_superclass import player_obj
class player_obj():
    def __init__(self, name):
        pass

from collections import deque
import numpy as np
from copy import copy

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Convolution2D, MaxPool2D, Flatten, LSTM, Reshape, GRU, Concatenate
from tensorflow.keras.optimizers import Adam


LEARNING_RATE = 0.001
ALPHA = 0.39  #  renew outputs
BATCH_SIZE=256
TIME_STEP_MEMORY=3

NUM_EPOCHS = 20

GAMMA = 0.96
REPLAY_MEMORY_SIZE = 3000
NUM_EPISODES = 10000
TARGET_UPDATE_FREQ = 100
MINIBATCH_SIZE = 1000

RANDOM_ACTION_DECAY = 0.99
INITIAL_RANDOM_ACTION = 1


# Main RL player
# input neural network model
# TODO: take parent from AI and write action function for RL
class player_RL(player_obj):
    def __init__(self, name):
        super(player_RL, self).__init__(name)
        # self.action_function = action_function
        self.replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)
        self.model = build_model()
        self.old_observation = (0, 0, 0)
        self.old_action = 0

    def done(self):
        super(player_RL, self).done()
        print(self.name, 'done')

    # TODO specify for RL
    def __str__(self):
        return self.id_game

    def move(self):
        self.old_observation = (self.env, self.data, self.reward)
        self.env, self.data, self.reward, self.info = self.connection.get_env_from_server(self.id_game)
        # saving data for training
        self.replay_buffer.add(self.old_observation[0], self.old_observation[1], self.old_action,
                               self.reward, self.env, self.data)

        # taking actions from model
        # From tank obj:  accelerate - 0{-1:1}, turn_body - 1{-1:1}, turn_tower - 2{-1:1}, shot - 3{Boolean}, skill - 4{Boolean}
        action, action_ind_for_save = self.action_function(self)
        self.old_action = copy(action_ind_for_save)

        self.connection.send_action(self.id_game, action)



    def action_function(self):
        # TODO add random
        # array (31, 1)
        preds = self.model.predict([self.env, self.data])[0]
        # decoding into [-1: 1] and Boolean
        return action_decoder(preds)


# TODO how to train a multiple choices? And what args to save?
def action_decoder(acts):
    args = [0 for _ in range(5)]
    args[0] = np.argmax(acts[:9])
    args[1] = np.argmax(acts[9:18])
    args[2] = np.argmax(acts[18:27])
    args[3] = np.argmax(acts[27:29])
    args[4] = np.argmax(acts[29:])
    return [Action_dict[args[0]], Action_dict[args[1]], Action_dict[args[2]], args[3], args[4]], args



# TODO each player have his folder with data

# save all actions and data into game object or may be each player will save it himself
class ReplayBuffer():

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque()

    # env + data = observation
    def add(self, env, data, action, reward, env_new, data_new):
        if len(self.buffer) > self.max_size:
            self.buffer.popleft()
        self.buffer.append((env, data, action, reward, env_new, data_new))

    # def sample(self, count):
    #     # taking a random part of buffer
    #     buffer_start = random.randint(0, len(self.buffer) - count)
    #     return deque(itertools.islice(self.buffer, buffer_start, buffer_start + count))

    def size(self):
        return len(self.buffer)


# function with NN model
def action_function(self):
    pass

# accelerate - 0{-1:1}, turn_body - 1{-1:1}, turn_tower - 2{-1:1}, shot - 3{Boolean}, skill - 4{Boolean}
#1: -> (9) [-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]
#2: -> (9) ...
#3: -> (9) ...
#4: -> (2) [0, 1]
#5: -> (2) ...
ACTIONS_DIM = 31
Action_dict = {x:np.linspace(-1, 1, 9)[x] for x in range(9)}

# data:  (10, 1) # 10: x, y, angle_tank, angle_tower, hp, speed, (time to reload: ammo, skill);
# ammunition; round time left in %
DATA_DIM = 10

def build_model():
    # map: (width, height, 4)
    # data:  (10, 1) # 10: x, y, angle_tank, angle_tower, hp, speed, (time to reload: ammo, skill);
              # ammunition; round time left in %

    input1 = tf.keras.layers.Input(shape=(15, 15, 4), name='map_env')
    input2 = tf.keras.layers.Input(shape=(DATA_DIM,), name='data')
    conv1 = Convolution2D(6, (3, 3), strides=(1), activation='relu')(input1)
    conv1 = MaxPool2D((2, 2))(conv1)
    conv2 = Convolution2D(8, (3, 3), strides=(1), activation='relu')(conv1)
    conv2 = MaxPool2D((2, 2))(conv2)
    fl = Flatten()(conv2)
    conc = Concatenate(axis=1)([fl, input2])
    dr = Dropout(0.5)(conc)
    denc = Dense(100, activation='relu')(dr)
    # (31)
    out = Dense(ACTIONS_DIM, activation='linear')(denc)

    model = tf.keras.Model(inputs=[input1, input2], outputs=out)

    # model.add(Reshape((1, -1)))  # , input_shape=(WEIGHT, HEIGHT, 2)))
    # model.add(GRU(100, return_sequences=False, stateful=True, reset_after=False))
    # model.add(LSTM(100, return_sequences=False, stateful=True))

    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss='mse',
        metrics=['accuracy'],
    )
    # print(model.summary())
    return model


# calculating rewards for train
def update_action(action_model, sample_transitions):
    # random.shuffle(sample_transitions)
    batch_observations = []
    batch_targets = []

    for sample_transition in sample_transitions:
        old_observation, action, reward, observation = sample_transition

        targets = get_q(action_model, old_observation)[0]
        next_step = 0
        if observation is not None:
            predictions = get_q(action_model, observation)
            new_action = np.argmax(predictions)
            next_step = GAMMA * predictions[0, new_action]

        targets[action] = (reward + next_step) * ALPHA + targets[action] * (1 - ALPHA)
        batch_observations.append(old_observation)
        batch_targets.append(targets)

    # prepare dataset for RNN
    batch_obs_rnn = make_trainset_for_rnn(batch_observations, TIME_STEP_MEMORY)

    return train(action_model, batch_obs_rnn, batch_targets)


def make_trainset_for_rnn(env_ds, time_step):
    time_step -= 1
    shapes = list(np.array(env_ds).shape)
    shapes.insert(1, time_step + 1)
    new_env = np.zeros(shapes)

    for i in range(len(env_ds)):
        if i < time_step:
            env_step = np.zeros(shapes[1:])
            env_step[time_step - i:time_step + 1] = env_ds[:i + 1]
            new_env[i] = env_step.copy()
        else:
            new_env[i] = env_ds[i - time_step:i + 1]
    return new_env


def train(model, observations, targets):
    # np_obs = np.reshape(observations, [-1, 1, OBSERVATIONS_DIM])
    np_obs = np.array(observations)
    np_targets = np.reshape(targets, [-1, ACTIONS_DIM])

    model.reset_states()  # need for LSTM
    acc = model.fit(np_obs, np_targets, epochs=NUM_EPOCHS, verbose=0, batch_size=BATCH_SIZE, shuffle=False)
    acc = int(acc.history['accuracy'][-1]*100)/100
    model.reset_states()  # need for LSTM

    return acc

# take action to move
def predict(model, observation):
    # np_obs = np.reshape(observation, [1, 1, OBSERVATIONS_DIM])
    np_obs = np.array([observation])

    action = np.argmax(model.predict(np_obs, batch_size=1))
    return action

# take outputs for training
def get_q(model, observation):
    # np_obs = np.reshape(observation, [1, 1, OBSERVATIONS_DIM])
    np_obs = np.array([observation])
    return model.predict(np_obs, batch_size=1)

if __name__ == '__main__':
    model = build_model()
    env = np.random.random((1, 15, 15, 4))
    data = np.random.random((1, 10))
    pred = model.predict([env, data])
    print(pred)
    print(action_decoder(pred[0]))

