import game.game as gg
import player.player as Player
import matplotlib.pyplot as plt


t_type = 'simple'



team1 = [Player.player_AI('AI1'), Player.player_AI("AI2")]
team1[0].change_tank_type(t_type); team1[1].change_tank_type(t_type)
team1[0].change_id(242); team1[1].change_id(243)

team2 = [Player.player_AI('AI3')]
team2[0].change_tank_type(t_type); team2[0].change_id(244)

print(team1[0])

Game = gg.TankGame()
Game.new_game(8, 12, team1, team2)

print('l', Game.team1[0].player.name)
team1[0].name = 'fff'
print('l', Game.team1[0].player.name)



plt.imshow(Game.collision_map, cmap='gray')


# plt.imshow(g.map[:,:, :3])
plt.show()




