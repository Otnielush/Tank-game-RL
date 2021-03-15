import game.game as gg
import player.player as pl
from net.broadcasting import connection
import matplotlib.pyplot as plt
from tank.tank_object import Tank

tt = Tank(1, 'simple')
print(tt)

g = gg.TankGame(6,10,5)
g.PIXELS_IN_CELL = 5
g.reset()
g.step()
g.sing()
connection = connection(2, True, (50,50,3), (10,))
plt.imshow(g.map[:,:, :3])
plt.show()

print(pl.player_RL)
# print(g.map[:,:,1])