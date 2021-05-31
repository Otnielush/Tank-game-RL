from numpy import argmax
from copy import deepcopy, copy
from random import randint
from os import path
import pandas as pd


if __name__ == '__main__': # for test
    class player_obj():
        def __init__(self, name):
            self.name = name
        def load_player(self):
            pass
else:
    from .player_superclass import player_obj

from collections import deque
import itertools
import numpy as np

# import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Convolution2D, MaxPool2D, Flatten, Concatenate, Input, \
    GlobalAveragePooling2D, LSTM, Reshape, GlobalMaxPool3D, GlobalMaxPool2D
from tensorflow.keras.optimizers import Adam
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

LEARNING_RATE = 0.001
BATCH_SIZE = 256

NUM_EPOCHS = 15

GAMMA = 0.94  # 0.96
TEAM_SPIRIT = 0.2
TIME_STEP_MEMORY = 8
REPLAY_MEMORY_SIZE = 5*60*20/2  # 3000

                # RL
RANDOM_ACTION_DECAY_FREQ = 40
ALPHA = 0.39  # renew outputs
ALPHA_DECAY = 0.01
INITIAL_RANDOM_ACTION = 90   # percent
RANDOM_ACTION_DECAY = 5

ENV_INPUT_2D_VIEW = (15, 15, 4)
# accelerate - 0{-1:1}, turn_body - 1{-1:1}, turn_tower - 2{-1:1}, shot - 3{Boolean}, skill - 4{Boolean}
NUM_SPEEDS = 7
# 1: -> (7) [-1.  , -0.66, -0.33,  0.,  0.33 ,  0.66,  1.  ]
# 2: -> (7) ...
# 3: -> (7) ...
# 4: -> (2) [0, 1]
# 5: -> (2) ...
ACTIONS_DIM = 25
Action_dict = {x: np.linspace(-1, 1, NUM_SPEEDS)[x] for x in range(NUM_SPEEDS)}
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
        self.prediction = 0
        self.old_prediction = 0
        self.random_move = INITIAL_RANDOM_ACTION  # percent of random choiced move
        self.games_to_rnd_decr = RANDOM_ACTION_DECAY_FREQ
        self.games_to_derc = self.games_to_rnd_decr
        # self.model = 0
        self.load_player()


    def done(self):
        super(player_RL, self).done()
        global ALPHA
        if self.random_move > 10:
            self.games_to_derc -= 1
        if self.games_to_derc <= 0:
            self.games_to_derc = self.games_to_rnd_decr
            self.random_move -= RANDOM_ACTION_DECAY
            ALPHA -= ALPHA_DECAY
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
        self.old_prediction = deepcopy(self.prediction)
        self.env, self.data, self.reward, self.info = self.connection.get_env_from_server(self.id_game)

        # taking actions from model
        # From tank obj:  accelerate - 0{-1:1}, turn_body - 1{-1:1}, turn_tower - 2{-1:1}, shot - 3{Boolean}, skill - 4{Boolean}
        action_indexes, action = self.action_function_RL()

        # saving data for training
        self.replay_buffer.add(self.old_observation, self.old_prediction, self.old_action, self.reward,
                               (deepcopy(self.env_2d_view), deepcopy(self.data)), deepcopy(self.prediction))
        self.old_action = copy(action_indexes)
        self.connection.send_action(self.id_game, action)


    def action_function_RL(self):
        self.env_prepare_2d()
        # array (25, 1)
        # RANDOM MOVE!
        if randint(0, 100) < self.random_move:
            args = [randint(0, 6), randint(7, 13), randint(14, 20), randint(21, 22), randint(23, 24)]
            return args, [Action_dict[args[0]], Action_dict[args[1]-NUM_SPEEDS], Action_dict[args[2]-NUM_SPEEDS*2],
                          args[3]-NUM_SPEEDS*3, args[4]-(NUM_SPEEDS*3+2)]
        # preparing 2d map view
        self.prediction = self.model.predict([self.env_2d_view, self.data], batch_size=1)[0]
        # decoding into [-1: 1] and Boolean
        return action_decoder(self.prediction)


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
            view = np.rot90(view, 2, (1, 2))
        elif self.start_side == 'right':
            view = np.rot90(view, 1, (1, 2))
        else:
            view = np.rot90(view, 3, (1, 2))

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
    args = [0 for _ in range(NUM_ACTIONS)]
    args[0] = argmax(acts[:7])
    args[1] = argmax(acts[7:14]) + 7
    args[2] = argmax(acts[14:21]) + 14
    args[3] = argmax(acts[21:23]) + 21
    args[4] = argmax(acts[23:]) + 23
    return args, [Action_dict[args[0]], Action_dict[args[1]-7], Action_dict[args[2]-14], args[3]-21, args[4]-23]

def action_decoder_reward(acts):
    args = np.zeros((NUM_ACTIONS), dtype='int32')
    args[0] = argmax(acts[:7])
    args[1] = argmax(acts[7:14]) + 7
    args[2] = argmax(acts[14:21]) + 14
    args[3] = argmax(acts[21:23]) + 21
    args[4] = argmax(acts[23:]) + 23
    rewards = np.zeros((ACTIONS_DIM))
    rewards[args] = acts[args]
    return rewards

def action_decoder_reward2(acts):
    reward = 0
    reward += acts[argmax(acts[:7])]
    reward += acts[argmax(acts[7:14]) + 7]
    reward += acts[argmax(acts[14:21]) + 14]
    reward += acts[argmax(acts[21:23]) + 21]
    reward += acts[argmax(acts[23:]) + 23]
    return reward


# save all actions and data into game object or may be each player will save it himself
class ReplayBuffer():

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque()

    # env + data = observation
    def add(self, observation, prediction, action, reward, observation_new, prediction_new):
        if len(self.buffer) > self.max_size:
            self.buffer.popleft()
        self.buffer.append((observation, prediction, action, reward, observation_new, prediction_new))

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

# https://stackoverflow.com/questions/44778439/keras-tf-time-distributed-cnnlstm-for-visual-recognition/47604360
# https://stackoverflow.com/questions/65148117/how-to-use-timedistributed-layer-with-concatenate-in-tensorflow
def build_model(for_train=False):
    # map: (15, 15, 4)
    # data:  (10, 1) # 10: x, y, angle_tank, angle_tower, hp, speed, (time to reload: ammo, skill);
    # ammunition; round time left in %

    # stateless
    if for_train:
        inp_shape = (TIME_STEP_MEMORY,)
        Statefull = False

        input1 = tf.keras.layers.Input(shape=inp_shape + ENV_INPUT_2D_VIEW, name='map_env')
        cnn = tf.keras.layers.Conv2D(6, 3, activation='relu')
        conv = tf.keras.layers.TimeDistributed(cnn)(input1)
        pool = tf.keras.layers.MaxPooling2D(3)
        conv = tf.keras.layers.TimeDistributed(pool)(conv)
        conc = tf.keras.layers.Reshape((TIME_STEP_MEMORY, -1))(conv)

        input2 = tf.keras.layers.Input(shape=inp_shape + (DATA_DIM,), name='data')

    # stateful
    else:
        inp_shape = (1,)
        Statefull = True
        input1 = tf.keras.layers.Input(batch_input_shape=inp_shape + ENV_INPUT_2D_VIEW, name='map_env')
        conv = tf.keras.layers.Conv2D(6, 3, activation='relu')(input1)
        conv = tf.keras.layers.MaxPooling2D(3)(conv)
        conc = tf.keras.layers.Reshape((1, -1))(conv)
        input2 = tf.keras.layers.Input(batch_input_shape=inp_shape + (1,) + (DATA_DIM,), name='data')

    conc = tf.keras.layers.Concatenate(axis=2)([conc, input2])
    denc = LSTM(30, return_sequences=False, stateful=Statefull)(conc)
    denc = Dense(40, activation='relu')(denc)
    # (25)
    out = Dense(ACTIONS_DIM, activation='linear')(denc)

    model = tf.keras.Model(inputs=[input1, input2], outputs=out)

    # model.add(Reshape((1, -1)))  # , input_shape=(WEIGHT, HEIGHT, 2)))
    # model.add(GRU(100, return_sequences=False, stateful=True, reset_after=False))
    # model.add(LSTM(100, return_sequences=False, stateful=True))

    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss='mae',
                metrics=['accuracy'],
    )
    # print(model.summary())
    #     tf.keras.utils.plot_model(model, "model1.png", show_shapes=False)
    return model


# 1. taking predictions from old observation
# 2. adding reward from next observation
# 3. training
# calculating rewards for train
def update_action_vect(action_model, sample_transitions, decoder=None, save_targets=False):
    buf_size = len(sample_transitions)-1
    ds_env_2d = np.zeros( (buf_size,) + ENV_INPUT_2D_VIEW )
    ds_data = np.zeros( (buf_size, DATA_DIM) )
    ds_actions = np.zeros( (buf_size, NUM_ACTIONS), dtype='int32')
    ds_targets = np.zeros( (buf_size, ACTIONS_DIM) )
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
        old_observation, old_prediction, actions, reward, observation, prediction = sample_transition

        ds_env_2d[i] = deepcopy(old_observation[0][0])
        ds_data[i] = deepcopy(old_observation[1][0])
        ds_actions[i] = deepcopy(actions)
        ds_rewards[i, actions] = reward / NUM_ACTIONS
        ds_targets[i] = deepcopy(old_prediction)
        if observation is not None:
            ds_env_2d_new[i] = deepcopy(observation[0][0])
            ds_data_new[i] = deepcopy(observation[1][0])
            ds_next_step[i] = deepcopy(prediction)
        i += 1

    print('. Reward calc.', end='')
    # ds_targets = action_model.predict([ds_env_2d, ds_data])
    # ds_next_step = action_model.predict([ds_env_2d_new, ds_data_new])

    # dataset check
    # for xx in ds_targets:
        # print(xx[:10].round(3))
    # save_rewards('RL1t1', ds_targets, 'targets_before')
    # save_rewards('RL1t1', ds_targets, 'targets_next')

    for i in range(len(ds_targets)):
        ds_targets[i, ds_actions[i]] *= (1 - ALPHA)
        ds_rewards[i, ds_actions[i]] += (action_decoder_reward2(ds_next_step[i]) / NUM_ACTIONS * GAMMA)  # return sum of 5 best rewards by moves

    ds_rewards = ds_rewards * ALPHA
    ds_targets = ds_targets + ds_rewards

    # save_rewards('RL1t1', ds_targets, 'targets_after')
    # prepare dataset for RNN
    # print('DS rnn 1', end='')
    ds_env_2d = make_trainset_for_rnn(ds_env_2d, TIME_STEP_MEMORY)
    # print(' 2.', end='')
    ds_data = make_trainset_for_rnn(ds_data, TIME_STEP_MEMORY)

    # print('P:',action_model.predict([ds_env_2d[0,:1], ds_data[0:1,:1]]), end=',')
    # model for training
    train_model = build_model(True)
    train_model.set_weights(action_model.get_weights())
    print('Training', end='')
    acc = train(train_model, [ds_env_2d, ds_data], ds_targets)
    action_model.set_weights(train_model.get_weights())
    # print('T:', action_model.predict([ds_env_2d[0, :1], ds_data[0:1, :1]]))
    return acc


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
    acc = model.fit(observations, np_targets, epochs=NUM_EPOCHS, verbose=0, shuffle=False)
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

# save rewards
def save_rewards(tank_name, data, file_name):
    folder = './/player//players data//' + tank_name + '//'
    df = pd.DataFrame(data, columns=[str(x+1) for x in range(31)])
    df.to_csv(folder+file_name+'.csv', index=None)
    del(df)

class Tankk():
    def __init__(self, x, y):
        self.X = x
        self.Y = y


if __name__ == '__main__':
    model = build_model()
    model.summary()
    model = build_model(True)
    model.summary()
    exit()
    pl = player_RL('RL1t1')
    pl.env = np.random.random((12, 12, 4))
    pl.tank_ingame = Tankk(2, 4)
    pl.start_side = 'up'
    pl.data = np.random.random((1,10))
    act, preds = pl.action_function_RL()

    print(preds)
    print(act)
    print(pl.data.shape)
    print(pl.env_2d_view.shape)


    ds_env_2d = np.random.random((10,15,15,4))
    ds_data = np.random.random((10,10))
    ds_targets = pl.model.predict([ds_env_2d, ds_data])
    print(ds_targets)




