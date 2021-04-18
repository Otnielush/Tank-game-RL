from .player_AI import player_AI
from .player_human import player_human
from .player_RL import player_RL


# import os
#
# dell = '/' if __name__ == '__main__' else '\\'
# print('name:', __name__)
#
# path = dell.join(__file__.split(dell)[:-1])+dell
# files = [f for f in os.listdir(path) if f.endswith('.py') and f != __file__.split(dell)[-1]]
#
# print(files)
#
# for f in files:
#     module_obj = __import__(path+f[:-3])
#     globals()[f[:-3]] = module_obj
#
# print(player_human_object)
#
# pp = player_human_object.player_human(3, 'ser')
#
# print(pp)