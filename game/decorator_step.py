from .game_object import TankGame
import numpy as np
from tank.ammo_object import Ammo

def step(self):
    data = self.connection.get_actions()

    # Move Tanks
    #  coll_map, accelerate, turn_body, turn_tower, shot, skill
    for i in range(len(self.team1)):
        if self.team1[i].hp > 0:
            idd = self.team1[i].id_game
            idd, old_yx, shot, skill = self.team1[i].move(self.map_coll, data[idd-self.ID_START])
            self.move_id_collision_map(self.team1[i], old_yx)  # changing collision map

            if shot:
                self.bullets.append(Ammo(self.team1[i], self.id_bul, self.team1[i].Y, self.team1[i].X, self.team1[i].direction_tank+self.team1[i].direction_tower))
                self.id_bul += 1  # id of bullets
    # TODO Stopped here

    for i in range(len(self.team2)):
        if self.team2[i].hp > 0:
            idd = self.team2[i].id_game
            self.team2[i].move(self.map_coll, data[idd - self.ID_START])

    # Move Bullets
    # TODO: move bullets


    # TODO function after
    self.calc_maps()
    return np.ones((self.height, self.width, 4))

setattr(TankGame, 'step', step)


def move_id_collision_map(self, obj, yx):
    self.map_coll[int(yx[0]*self.PIX_CELL): int((yx[0]+1)*self.PIX_CELL), int(yx[1]*self.PIX_CELL): int((yx[1]+1)*self.PIX_CELL), 1] = 0
    self.map_coll[int(obj.Y * self.PIX_CELL): int((obj.Y + 1) * self.PIX_CELL), int(obj.X * self.PIX_CELL): int((obj.X + 1) * self.PIX_CELL), 1] = obj.id_game




setattr(TankGame, 'move_id_collision_map', move_id_collision_map)