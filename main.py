import game.game as gg
import player.player as Player
import matplotlib.pyplot as plt
from options.video import WIDTH, HEIGHT, MULTY_PIXEL, FRAME_RATE
import time



VIDEO = [True]





t_type = 'simple'

team1 = [Player.player_AI('AI1')] #  , Player.player_AI("AI2")]
team1[0].change_tank_type(t_type); team1[0].change_id(242)
# team1[1].change_tank_type(t_type); team1[1].change_id(243)

team2 = [Player.player_RL('AI3'), Player.player_AI('AI4'), Player.player_AI('AI5')]
team2[0].change_tank_type(t_type); team2[0].change_id(244)
team2[1].change_tank_type(t_type); team2[1].change_id(245)
team2[2].change_tank_type(t_type); team2[2].change_id(246)


Game = gg.TankGame(MULTY_PIXEL)
Game.new_game(WIDTH, HEIGHT, team1, team2, VIDEO)



# accelerate - 0, turn_body - 1, turn_tower - 2, shot - 3, skill - 4
Game.connection.send_action(101, [1, 1, 0.5, True, False])
Game.connection.send_action(102, [0.1, 0.1, 0, False, False])
Game.connection.send_action(103, [0.0, -1, 0, False, False])
Game.connection.send_action(104, [0.1, 0.0, 0, False, False])

time_start = time.time()
frame = 1

if VIDEO[0]:
    from video import graphics
    graphics.video_build_map(Game)

done = False
while True:


    # stop game check TODO change to net connection so that the game and the players would be a different programs
    if done:
        # for training,
        for t in Game.id_tanks:
            Game.id_tanks[t].player.done()
        break
    # players decisions to move
    elif Game.frame_step <= 0:
        for t in Game.id_tanks:
            Game.id_tanks[t].player.move()
            pass


    done = Game.step()

    if VIDEO[0]:
        graphics.play_video(Game, VIDEO)

    print('\rspeed:', round(Game.team1[0].speed, 3), 'FPS:', round(frame/(time.time()-time_start), 1),
          'xy:',round(Game.team1[0].X,1), round(Game.team1[0].Y,1), end=' |')
    # print('bullets fired:', len(Game.bullets), 'in fly:', len(Game.bullets_in_act))


    frame += 1





