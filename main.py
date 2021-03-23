import game.game as gg
import player.player as Player
import matplotlib.pyplot as plt


t_type = 'simple'

team1 = [Player.player_AI('AI1')] #  , Player.player_AI("AI2")]
team1[0].change_tank_type(t_type); team1[0].change_id(242)
# team1[1].change_tank_type(t_type); team1[1].change_id(243)


team2 = [Player.player_AI('AI3'), Player.player_AI('AI4'), Player.player_AI('AI5')]
team2[0].change_tank_type(t_type); team2[0].change_id(244)
team2[1].change_tank_type(t_type); team2[1].change_id(245)
team2[2].change_tank_type(t_type); team2[2].change_id(246)

print(team1[0])

Game = gg.TankGame(10)
Game.new_game(10, 12, team1, team2)


Game.team1[0].max_speed = 2
Game.team1[0].direction_tank = 0.15
print(Game.team1[0].direction_tank)

print('coords', Game.team1[0].Y, Game.team1[0].X)
plt.imshow(Game.map_env[:,:,:3])
plt.show()

# accelerate - 0, turn_body - 1, turn_tower - 2, shot - 3, skill - 4
Game.connection.send_action(0, [1, 0.0, 1, False, False])
Game.step()


print('coords', Game.team1[0].Y, Game.team1[0].X)
plt.imshow(Game.map_env[:,:,:3])
plt.show()

Game.step()


print('coords', Game.team1[0].Y, Game.team1[0].X)
plt.imshow(Game.map_env[:,:,:3])
plt.show()
# plt.imshow(Game.map_env[:,:,:3])
# plt.imshow(g.map[:,:, :3])
# plt.show()




