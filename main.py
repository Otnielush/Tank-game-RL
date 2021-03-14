import game.game as gg
import player.player as pl
from net.broadcasting import connection


g = gg.TankGame(1,2)
g.step()
g.sing()
connection = connection(2, True, (50,50,3), (10,))


print(pl.player_RL)
print(connection)