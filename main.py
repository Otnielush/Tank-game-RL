import game.game as gg
import player.player as pl
from net.broadcasting import connection
import matplotlib.pyplot as plt
from tank.tank_object import Tank
from tank.ammo_object import Ammo

import numpy as np



tt = Tank(1, 'simple', 5,5)

print(tt)
aa = Ammo(tt, 1,  0, 0, 180)
print(aa)
aa.move()
aa.move()
print(aa)
print(aa.hit())


g = gg.TankGame(6,10,5)
plt.imshow(g.collision_map, cmap='gray')
g.PIXELS_IN_CELL = 5
g.reset()
g.step()
g.sing()
connection = connection(2, True, (50,50,3), (10,))
# plt.imshow(g.map[:,:, :3])
plt.show()




