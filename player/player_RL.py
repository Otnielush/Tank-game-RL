from numpy import argmax
from copy import deepcopy, copy
from random import randint
from os import path
import pandas as pd


if __name__ == '__main__': # for test
    class player_obj():
        def __init__(self, name):
            pass
else:
    from .player_superclass import player_obj

from collections import deque
import itertools
import numpy as np

# import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Convolution2D, MaxPool2D, Flatten, Concatenate, Input
from tensorflow.keras.optimizers import Adam
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

LEARNING_RATE = 0.001
ALPHA = 0.39  # renew outputs
BATCH_SIZE = 256
TIME_STEP_MEMORY = 3

NUM_EPOCHS = 20

GAMMA = 0.96
REPLAY_MEMORY_SIZE = 5*60*20/2  # 3000

                # RL #
RANDOM_ACTION_DECAY_FREQ = 10
RANDOM_ACTION_DECAY = 5
INITIAL_RANDOM_ACTION = 10   # percent

ENV_INPUT_2D_VIEW = (15, 15, 4)
# accelerate - 0{-1:1}, turn_body - 1{-1:1}, turn_tower - 2{-1:1}, shot - 3{Boolean}, skill - 4{Boolean}
# 1: -> (9) [-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]
# 2: -> (9) ...
# 3: -> (9) ...
# 4: -> (2) [0, 1]
# 5: -> (2) ...
ACTIONS_DIM = 31
Action_dict = {x: np.linspace(-1, 1, 9)[x] for x in range(9)}
NUM_ACTIONS = 5

# data:  (10, 1) # 10: x, y, angle_tank, angle_tower, hp, speed, (time to reload: ammo, skill);
# ammunition; round time left in %
DATA_DIM = 10


# Main RL player
# input neural network model
# TODO: take parent from AI and write action function for RL
class player_RL(player_obj):
    def __init__(self, name):
        super(player_RL, self).__init__(name)
        # self.action_function = action_function
        self.replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)
        self.old_observation = (0, 0, 0)
        self.env_2d_view = 0
        self.old_action = 0
        self.random_move = INITIAL_RANDOM_ACTION  # percent of random choiced move
        self.games_to_rnd_decr = RANDOM_ACTION_DECAY_FREQ
        self.games_to_derc = self.games_to_rnd_decr
        # self.model = 0
        self.load_player()


    def done(self):
        super(player_RL, self).done()
        if self.random_move > 10:
            self.games_to_derc -= 1
        if self.games_to_derc <= 0:
            self.games_to_derc = self.games_to_rnd_decr
            self.random_move -= RANDOM_ACTION_DECAY
            print(self.name, 'random move:', self.random_move)
        print(self.name, 'game done.', end='')
        # training
        sample_transitions = self.replay_buffer.get_buffer()
        save_targets = self.name if self.id_game == 101 else False
        acc = update_action_vect(self.model, sample_transitions, decoder=action_decoder, save_targets=save_targets)
        print(' accuracy:', acc)
        self.replay_buffer.clear()
        self.save_model()


    # TODO specify for RL
    # def __str__(self):
    #     super(player_RL, self).__str__()


    def move(self):
        self.old_observation = [deepcopy(self.env_2d_view), deepcopy(self.data)]
        self.env, self.data, self.reward, self.info = self.connection.get_env_from_server(self.id_game)

        # taking actions from model
        # From tank obj:  accelerate - 0{-1:1}, turn_body - 1{-1:1}, turn_tower - 2{-1:1}, shot - 3{Boolean}, skill - 4{Boolean}
        action_indexes, action = self.action_function_RL()

        # saving data for training
        self.replay_buffer.add(self.old_observation, self.old_action, self.reward, (deepcopy(self.env_2d_view), deepcopy(self.data)))
        self.old_action = copy(action_indexes)
        self.connection.send_action(self.id_game, action)


    def action_function_RL(self):
        self.env_prepare_2d()
        # array (31, 1)
        # RANDOM MOVE!
        if randint(0, 100) < self.random_move:
            args = [randint(0, 8), randint(9, 17), randint(18, 26), randint(27, 28), randint(29, 30)]
            return args, [Action_dict[args[0]], Action_dict[args[1]-9], Action_dict[args[2]-18], args[3]-27, args[4]-29]
        # preparing 2d map view
        predictions = self.model.predict([self.env_2d_view, self.data], batch_size=1)[0]
        # decoding into [-1: 1] and Boolean
        return action_decoder(predictions)


    # need: env_map, position of tank (x,y), starting side of map
    def env_prepare_2d(self):
        view = np.zeros((1,) + ENV_INPUT_2D_VIEW)

        start_x = max(0, 7 - int(self.tank_ingame.X))  # 7 - center of view ENV_INPUT_2D_VIEW = (15, 15, 4)
        start_y = max(0, 7 - int(self.tank_ingame.Y))
        start_x_e = max(0, int(self.tank_ingame.X) - ENV_INPUT_2D_VIEW[0] // 2)
        start_y_e = max(0, int(self.tank_ingame.Y) - ENV_INPUT_2D_VIEW[1] // 2)

        range_x = min(self.env.shape[0] - start_x_e, 15 - start_x)  # 15 shape of view
        range_y = min(self.env.shape[1] - start_y_e, 15 - start_y)

        view[0, start_x: start_x + range_x, start_y: start_y + range_y, :] = \
            deepcopy(self.env[start_x_e: start_x_e + range_x, start_y_e: start_y_e + range_y, :])

        # rotation for 2d view
        if self.start_side == 'up':
            pass
        elif self.start_side == 'down':
            view = np.rot90(view, 2)
        elif self.start_side == 'right':
            view = np.rot90(view)
        else:
            view = np.rot90(view, 3)

        self.env_2d_view = view


    def load_player(self):
        super(player_RL, self).load_player()
        self.model = build_model()
        # check folder for saves
        folder = './/player//players data//'+self.name+'//model_2d'
        if path.exists(folder+'.index'):
        # if path.exists(folder):
            self.model.load_weights(folder)
            print(self.name, 'model loaded')


    # saving model weights to directory named by player name
    def save_model(self):
        super(player_RL, self).save_model()
        folder = './/player//players data//' + self.name + '//model_2d'
        self.model.save_weights(folder)
        # self.model.save(folder)



# from outputs of NN we take 3 choises
def action_decoder(acts):
    args = [0 for _ in range(5)]
    args[0] = argmax(acts[:9])
    args[1] = argmax(acts[9:18]) + 9
    args[2] = argmax(acts[18:27]) + 18
    args[3] = argmax(acts[27:29]) + 27
    args[4] = argmax(acts[29:]) + 29
    return args, [Action_dict[args[0]], Action_dict[args[1]-9], Action_dict[args[2]-18], args[3]-27, args[4]-29]

def action_decoder_reward(acts):
    args = np.zeros((NUM_ACTIONS), dtype='int32')
    args[0] = argmax(acts[:9])
    args[1] = argmax(acts[9:18]) + 9
    args[2] = argmax(acts[18:27]) + 18
    args[3] = argmax(acts[27:29]) + 27
    args[4] = argmax(acts[29:]) + 29
    rewards = np.zeros((ACTIONS_DIM))
    rewards[args] = acts[args]
    return rewards

def action_decoder_reward2(acts):
    reward = 0
    reward += acts[argmax(acts[:9])]
    reward += acts[argmax(acts[9:18]) + 9]
    reward += acts[argmax(acts[18:27]) + 18]
    reward += acts[argmax(acts[27:29]) + 27]
    reward += acts[argmax(acts[29:]) + 29]
    return reward


# save all actions and data into game object or may be each player will save it himself
class ReplayBuffer():

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque()

    # env + data = observation
    def add(self, observation, action, reward, observation_new):
        if len(self.buffer) > self.max_size:
            self.buffer.popleft()
        self.buffer.append((observation, action, reward, observation_new))

    # def sample(self, count):
    #     # taking a random part of buffer
    #     buffer_start = random.randint(0, len(self.buffer) - count)
    #     return deque(itertools.islice(self.buffer, buffer_start, buffer_start + count))

    def size(self):
        return len(self.buffer)

    def get_buffer(self):
        return deque(itertools.islice(self.buffer, 0, len(self.buffer)))

    def clear(self):
        self.buffer = deque()


def build_model():
    # map: (15, 15, 4)
    # data:  (10, 1) # 10: x, y, angle_tank, angle_tower, hp, speed, (time to reload: ammo, skill);
    # ammunition; round time left in %

    input1 = Input(shape=ENV_INPUT_2D_VIEW, name='map_env')
    input2 = Input(shape=(DATA_DIM,), name='data')
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

# 1. taking predictions from old observation
# 2. adding reward from next observation
# 3. training
# calculating rewards for train
def update_action(action_model, sample_transitions, decoder=None):
    buf_size = len(sample_transitions)-1
    env_2d_obs = np.zeros( (buf_size,) + ENV_INPUT_2D_VIEW )
    data_obs = np.zeros( (buf_size, DATA_DIM) )
    batch_targets = []
    first = True
    i = 0
    print(' Data prep: 00%', end='')
    for sample_transition in sample_transitions:  # first is 0
        print('\b\b\b\b{:3.0f}%'.format(i/buf_size*100), end='')
        if first:
            first = False
            continue
        old_observation, actions, reward, observation = sample_transition

        # TODO stopped here. Training
        targets = get_q(action_model, old_observation)[0]
        next_step = 0
        if observation is not None:
            predictions = get_q(action_model, observation)[0]
            # for complex models
            if decoder:
                new_action, _ = decoder(predictions)
                next_step = GAMMA * sum(predictions[new_action])
            else:
                new_action = np.argmax(predictions)
                next_step = GAMMA * predictions[new_action]

        targets[actions] = ((reward + next_step) / NUM_ACTIONS) * ALPHA + targets[actions] * (1 - ALPHA)
        env_2d_obs[i] = deepcopy(old_observation[0][0])
        data_obs[i] = deepcopy(old_observation[1][0])
        batch_targets.append(targets)
        i += 1
    print('. Training', end='')
    # prepare dataset for RNN
    # batch_obs_rnn = make_trainset_for_rnn(batch_observations, TIME_STEP_MEMORY)

    return train(action_model, [env_2d_obs, data_obs], batch_targets)

def update_action_vect(action_model, sample_transitions, decoder=None, save_targets=False):
    buf_size = len(sample_transitions)-1
    ds_env_2d = np.zeros( (buf_size,) + ENV_INPUT_2D_VIEW )
    ds_data = np.zeros( (buf_size, DATA_DIM) )
    ds_actions = np.zeros( (buf_size, NUM_ACTIONS), dtype='int32')
    # ds_targets = np.zeros( (buf_size, ACTIONS_DIM) )
    ds_rewards = np.zeros( (buf_size, ACTIONS_DIM) )
    ds_next_step = np.zeros( (buf_size, ACTIONS_DIM) )
    ds_env_2d_new = np.zeros( (buf_size,) + ENV_INPUT_2D_VIEW )
    ds_data_new = np.zeros( (buf_size, DATA_DIM) )

    first = True
    i = 0
    print(' Data prep: 00%', end='')
    # Vectorising data
    for sample_transition in sample_transitions:  # first is 0
        print('\b\b\b\b{:3.0f}%'.format(i/buf_size*100), end='')
        if first:
            first = False
            continue
        old_observation, actions, reward, observation = sample_transition

        ds_env_2d[i] = deepcopy(old_observation[0][0])
        ds_data[i] = deepcopy(old_observation[1][0])
        ds_actions[i] = deepcopy(actions)
        ds_rewards[i, actions] = reward / NUM_ACTIONS
        if observation is not None:
            ds_env_2d_new[i] = deepcopy(observation[0][0])
            ds_data_new[i] = deepcopy(observation[1][0])
        i += 1

    print('. Reward calc.', end='')
    ds_targets = action_model.predict([ds_env_2d, ds_data])
    ds_next_step = action_model.predict([ds_env_2d_new, ds_data_new])

    for i in range(len(ds_targets)):
        ds_targets[i, ds_actions[i]] *= (1 - ALPHA)
        ds_rewards[i, ds_actions[i]] += (action_decoder_reward2(ds_next_step[i]) / NUM_ACTIONS * GAMMA)  # return sum of 5 best rewards by moves

    ds_rewards = ds_rewards * ALPHA
    ds_targets = ds_targets + ds_rewards

    print('Training', end='')
    # prepare dataset for RNN
    # batch_obs_rnn = make_trainset_for_rnn(batch_observations, TIME_STEP_MEMORY)

    return train(action_model, [ds_env_2d, ds_data], ds_targets)


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
    # np_obs = np.array(observations)

    np_targets = np.reshape(targets, [-1, ACTIONS_DIM])

    # model.reset_states()  # need for LSTM
    acc = model.fit(observations, np_targets, epochs=NUM_EPOCHS, verbose=0, batch_size=BATCH_SIZE, shuffle=False)
    acc = int(acc.history['accuracy'][-1] * 100) / 100
    # model.reset_states()  # need for LSTM

    return acc


# take action to move
def predict(model, observation):
    # np_obs = np.reshape(observation, [1, 1, OBSERVATIONS_DIM])
    np_obs = np.array([observation])

    action = np.argmax(model.predict(np_obs, batch_size=1))
    return action


# take outputs for training
def get_q(model, observation):
    return model.predict(observation, batch_size=1)


class Tankk():
    def __init__(self, x, y):
        self.X = x
        self.Y = y


if __name__ == '__main__':
    # model = build_model()
    tt = np.random.random((31))
    print(tt)
    print(action_decoder_reward(tt))

    exit()
    pl = player_RL('ttt')
    pl.env = np.random.random((12, 10, 4))
    pl.tank_ingame = Tankk(2, 4)
    pl.start_side = 'up'
    pl.data = np.random.random((1,10))
    act, preds = pl.action_function_RL()

    print(preds)
    print(act)
    print(pl.data.shape)
    print(pl.env_2d_view.shape)
    print(pl.data)

