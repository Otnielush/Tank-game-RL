# TODO: STRUCTURE: I build with many files so we can work and not to interrupt each other
#  game i did by decorators so we can make a few versions of of game


# TODO: Questions
#  1. now 2 layers on map for friendly team and enemies, Make: 1 layer enemies with "minus" coefficients.
#  So for another team enough *-1


# TODO: how to code net games? Ask one after another players to move? Or make broadcasting envinronment
#  to all players and receive actions from them. Save received actions for each player and do this action
#  if changed action as do newer. Or do nothing untill action received. We can broadcast to Global variables

# TODO DONE: "net connection" - broadcasting

# TODO: How to make collisions? I decided to make collision map with ids on coordinates. If value of cell more 0,
#  so there is object there with id of value of cell. When object moving he change place of his ids on map.

# TODO: in game 2 maps: 1. for game itself (moving, collision, video) 2. for input AI players
# TODO: for types of tanks and obstacles i ranged number from boolean (0,1) to float (0, 0.1,...0.9, 1)

# TODO: 1. Make visiulization for game by pygame (doing now)
# TODO: 2. game engine - ammo hit
# TODO: add aim for rocket by AI
# TODO: 3. May be create log for all actions in game and on its basis give rewards






# TODO Algorithms to use:

# No-regret learning is a natural extension of reinforcement learning. Both forms of learning are based on payoffs in past play.
# The difference between reinforcement learning and no-regret learning is that reinforcement learning is based on realised play,
# while no-regret learning is based on hypothetical play (i.e., what could have been played). A requirement for no-regret learning is
# that alternative payoffs are known, for example through complete knowledge of a stationary payoff matrix.
#
# “Strategic Learning and its Limits” Peyton Young (2004). Ch. 2
# “Learning and Teaching” Shoham (2009). Ch. 7 of Multi-agent Systems
# “Regret in the On-Line Decision Problem” Foster et al. (1999)
# “A Simple Adaptive Procedure Leading to Correlated Equilibrium” Hart et al. (2000)
# “Convergence and No-Regret in MAL” M. Bowling (2005)

# Reinforcement Learning
# https://gym.openai.com/
# https://www.pygame.org/news
# https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://medium.com/deep-math-machine-learning-ai/ch-13-deep-reinforcement-learning-deep-q-learning-and-policy-gradients-towards-agi-a2a0b611617e
# https://www.mlq.ai/deep-reinforcement-learning-q-learning/
# https://rubikscode.net/2019/07/08/deep-q-learning-with-python-and-tensorflow-2-0/

