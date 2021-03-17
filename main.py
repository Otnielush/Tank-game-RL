import game.game as gg
import player.player as Player
from net.broadcasting import net_connection
import matplotlib.pyplot as plt
from tank.tank_object import Tank
from tank.ammo_object import Ammo

import numpy as np
t_type = 'simple'
team1 = [Player.player_AI('AI1').change_tank_type(t_type).change_id(242), Player.player_AI("AI2").change_tank_type(t_type).change_id(243)]
team2 = [Player.player_AI('AI3').change_tank_type(t_type).change_id(244)]

Game = gg.TankGame()
Game.new_game(8, 12, team1, team2)

tt = Tank(1,'ff', 'simple', 5,5)

pl1 = pl.player_AI('vasya')

g = gg.TankGame()
plt.imshow(g.collision_map, cmap='gray')
g.PIXELS_IN_CELL = 5
g.reset()
g.step()

connection = net_connection(2, True, (50, 50, 3), (10,))
# plt.imshow(g.map[:,:, :3])
plt.show()




