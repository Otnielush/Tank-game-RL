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

Game = gg.TankGame(5)
Game.new_game(10, 12, team1, team2)


Game.team1[0].direction_tank = 0.05
print(Game.team1[0].direction_tank)

print('speed', Game.team1[0].reloading_ammo, Game.team1[0].speed)
plt.imshow(Game.map_coll[:,:,:3])
plt.show()

# accelerate - 0, turn_body - 1, turn_tower - 2, shot - 3, skill - 4
Game.connection.send_action(0, [1, 0.0, 0.5, True, False])
Game.connection.send_action(1, [1, 0.5, 0, False, False])
Game.connection.send_action(2, [1, -0.5, 0, False, False])
Game.connection.send_action(3, [1, 0.0, 0, False, False])
Game.step()


print('speed', Game.team1[0].reloading_ammo, Game.team1[0].speed)
plt.imshow(Game.map_coll[:,:,:3])
plt.show()

Game.step()


print('speed', Game.team1[0].reloading_ammo, Game.team1[0].speed)
plt.imshow(Game.map_coll[:,:,:3])
plt.show()
# plt.imshow(Game.map_env[:,:,:3])
# plt.imshow(g.map[:,:, :3])
# plt.show()




