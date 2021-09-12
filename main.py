import game.game as gg
import player.player as Player
import matplotlib.pyplot as plt
from options.video import WIDTH, HEIGHT, MULTY_PIXEL, FRAME_RATE
import time
from video import graphics


VIDEO = [True]
ROUNDS = 50
VIDEO_ROUNDS = [0, 1001]




t_type = 'laser'  #'simple','laser'

team1 = []
team1 = [Player.player_RL("RL1t1ml")]; team1[0].change_tank_type(t_type); team1[0].change_id(242)
team1.append(Player.player_RL('RL2t1ml')); team1[1].change_tank_type('simple'); team1[1].change_id(243)
# team1.append(Player.player_RL('RL3t1')); team1[2].change_tank_type(t_type); team1[2].change_id(244)
# team1.append(Player.player_AI('Bob')); team1[2].change_tank_type('simple'); team1[2].change_id(1000)

team2 = [Player.player_RL("RL1t2ml")]; team2[0].change_tank_type(t_type); team2[0].change_id(245)
team2.append(Player.player_RL('RL2t2ml')); team2[1].change_tank_type('simple'); team2[1].change_id(246)
# team2.append(Player.player_RL('RL3t2')); team2[2].change_tank_type(t_type); team2[2].change_id(247)



Game = gg.TankGame(MULTY_PIXEL)
game_round = 1
GAME_TYPE = 'shooting'  # 'shooting'
Game.new_game(WIDTH, HEIGHT, team1, team2, VIDEO, type_m=GAME_TYPE)
# Game.time_round_len = FRAME_RATE*30
print('_______ Round 1 _______')

# FOR TEST
# accelerate - 0, turn_body - 1, turn_tower - 2, shot - 3, skill - 4
# Game.connection.send_action(101, [1, 1, 0.5, True, False])

time_start = time.time()
frame = 1

if VIDEO[0]:
    graphics.init_display(Game.width, Game.height)
    graphics.video_build_map(Game)

import tensorflow as tf
tf.keras.utils.plot_model(Game.team1[0].player.model, "model2.png", show_shapes=True)


done = False
while True:

    # stop game check TODO change to net connection so that the game and the players would be a different programs
    if done:
        print()
        # saving history of rewards
        # Game.rewards_to_csv(team1[0].id_game, 'rewards')
        # for training,
        for t in Game.id_tanks:
            # Game.rewards_to_csv(Game.team1[0].id_game, 'game_rewards')
            Game.id_tanks[t].player.done()

        # new Game game_round
        game_round += 1
        if game_round > ROUNDS:
            print(Game.team1[0].player)
            break
        if game_round in VIDEO_ROUNDS:
            VIDEO[0] = True
        else:
            VIDEO[0] = False
        time_start = time.time()
        frame = 1
        # dummies moving after half of rounds
        if game_round // 2 == ROUNDS:
            Game.shooting_moving = True
        Game.new_game(WIDTH, HEIGHT, team1, team2, VIDEO, type_m=GAME_TYPE)
        if VIDEO[0]:
            graphics.init_display(Game.width, Game.height)
            graphics.video_build_map(Game)
        else:
            graphics.close_window()
        print('\r_______ Round', game_round, '_______')

    # players decisions to move
    elif Game.frame_step <= 0:
        for t in Game.id_tanks:
            Game.id_tanks[t].player.move()


    done = Game.step()

    if VIDEO[0]:
        graphics.play_video(Game, VIDEO)

    print('\rFPS', round(frame/(time.time()-time_start+0.1), 1), end='')
    # for ii in range(len(Game.team1)):
    #     print(Game.team1[ii].id_game, Game.team1[ii].name)

    # print('\rspeed:', round(Game.team1[0].speed, 3), 'FPS:', round(frame/(time.time()-time_passed), 1),
          # 'xy:',round(Game.team1[0].X,1), round(Game.team1[0].Y,1), end=' |')
    # print('bullets fired:', len(Game.bullets), 'in fly:', len(Game.bullets_in_act))


    frame += 1





