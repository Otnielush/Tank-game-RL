import pygame
from options.video import WIDTH, HEIGHT, MULTY_PIXEL, MOVES_PER_FRAME, FRAME_RATE
import numpy as np
import os
import time

MULTY_PIXEL *= 10

# run display
pygame.init()
clock = pygame.time.Clock()

DISPLAY = pygame.display.set_mode((WIDTH * MULTY_PIXEL *2, HEIGHT * MULTY_PIXEL))

# background TODO: add different grounds
background = pygame.surface.Surface((WIDTH * MULTY_PIXEL, HEIGHT * MULTY_PIXEL))
background.fill((50, 50, 50))
color_lines = (250,250,100)
for x in np.arange(1, WIDTH):
    pygame.draw.lines(background, color_lines, False, [(x * MULTY_PIXEL, 0), (x * MULTY_PIXEL, HEIGHT * MULTY_PIXEL)], 1)
for y in np.arange(1, HEIGHT):
    pygame.draw.lines(background, color_lines, False, [(0, y * MULTY_PIXEL), (WIDTH * MULTY_PIXEL, y * MULTY_PIXEL)], 1)

pygame.display.update()
pygame.display.set_caption("Tank game RL")

# colors


img_tank_base = pygame.transform.scale(pygame.image.load(os.path.join('video', 'pics', 'tank2.png')), (round(0.6*MULTY_PIXEL), 1*MULTY_PIXEL))
img_tank_tower = pygame.image.load(os.path.join('video', 'pics', 'tank_tower.png'))



def play_video(game):
    game_end = False
    move = 0
    turn = 0
    tower = 0
    shot = False
    coll_map = pygame.transform.rotate(pygame.surfarray.make_surface(game.map_coll * 255), 0)
    coll_map = pygame.transform.scale(coll_map, (WIDTH * MULTY_PIXEL, HEIGHT * MULTY_PIXEL))

    # features_nn = pygame.transform.rotate(pygame.surfarray.make_surface(game.map_env[:, :, :3] * 255), 0)
    # features_nn = pygame.transform.scale(features_nn, (WIDTH * MULTY_PIXEL, HEIGHT * MULTY_PIXEL))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_end = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                move = 1
            if event.key == pygame.K_DOWN:
                move = -1
            if event.key == pygame.K_LEFT:
                turn = 1
            if event.key == pygame.K_RIGHT:
                turn = -1
            if event.key == pygame.K_RCTRL or event.key == pygame.K_LCTRL:
                shot = True
            if event.key == pygame.K_a:
                tower = 1
            if event.key == pygame.K_d:
                tower = -1

        game.connection.send_action(0, [move, turn, tower, shot, False])

    DISPLAY.blit(coll_map, (DISPLAY.get_width() // 2, 0))
    # DISPLAY.blit(features_nn, (0, DISPLAY.get_height() // 2))

    DISPLAY.blit(background, (0, 0))
    for tank in game.team1:
        DISPLAY.blit(pygame.transform.rotate(img_tank_base, tank.direction_tank*360), (tank.X*MULTY_PIXEL, tank.Y*MULTY_PIXEL))

    # for tank in game.team2:
    #     DISPLAY.blit(pygame.transform.rotate(img_tank_base, (tank.direction_tank) * 360),
    #                  (tank.X * MULTY_PIXEL, tank.Y * MULTY_PIXEL))



    pygame.display.update()
    time.sleep(1/FRAME_RATE)

