import game.game as gg
import player.player as Player
import matplotlib.pyplot as plt
from options.video import WIDTH, HEIGHT, MULTY_PIXEL, FRAME_RATE
import time
from video import graphics



VIDEO = [False]
ROUNDS = 10
VIDEO_ROUNDS = [1, 10]





t_type = 'simple'

team1 = [Player.player_RL("RL1t1"), Player.player_RL('RL2t1')]
team1[0].change_tank_type(t_type); team1[0].change_id(242)
team1[1].change_tank_type(t_type); team1[1].change_id(243)

team2 = [Player.player_RL('RL1t2'), Player.player_RL('RL2t2'), Player.player_RL('RL3t2')]
team2[0].change_tank_type(t_type); team2[0].change_id(244)
team2[1].change_tank_type(t_type); team2[1].change_id(245)
team2[2].change_tank_type(t_type); team2[2].change_id(246)


Game = gg.TankGame(MULTY_PIXEL)
game_round = 1
Game.new_game(WIDTH, HEIGHT, team1, team2, VIDEO)
# Game.time_round_len = 2
print('_______ Round 1 _______')

# FOR TEST
# accelerate - 0, turn_body - 1, turn_tower - 2, shot - 3, skill - 4
# Game.connection.send_action(101, [1, 1, 0.5, True, False])

time_start = time.time()
frame = 1

graphics.video_build_map(Game)

done = False
while True:


    # stop game check TODO change to net connection so that the game and the players would be a different programs
    if done:
        print()
        # for training,
        for t in Game.id_tanks:
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
        Game.new_game(WIDTH, HEIGHT, team1, team2, VIDEO)
        print('\r_______ Round', game_round, '_______')
        time_start = time.time()
        frame = 1

    # players decisions to move
    elif Game.frame_step <= 0:
        for t in Game.id_tanks:
            Game.id_tanks[t].player.move()


    done = Game.step()

    if VIDEO[0]:
        graphics.play_video(Game, VIDEO)

    print('\rFPS', round(frame/(time.time()-time_start), 1), end='')

    # print('\rspeed:', round(Game.team1[0].speed, 3), 'FPS:', round(frame/(time.time()-time_start), 1),
          # 'xy:',round(Game.team1[0].X,1), round(Game.team1[0].Y,1), end=' |')
    # print('bullets fired:', len(Game.bullets), 'in fly:', len(Game.bullets_in_act))


    frame += 1





