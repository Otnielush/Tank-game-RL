import pygame
from options.video import WIDTH, HEIGHT, MULTY_PIXEL, MOVES_PER_FRAME, FRAME_RATE
import numpy as np
import os
import time

MULTY_PIXEL = 50

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

# TODO change to scaling by type tank
tower_width = 0.4*MULTY_PIXEL
tower_height = 1*MULTY_PIXEL
tank_width = round(0.6*MULTY_PIXEL)
tank_height =  1*MULTY_PIXEL
img_tank_base = pygame.transform.scale(pygame.image.load(os.path.join('video', 'pics', 'tank.png')), (tank_width, tank_height))
img_tank_tower = pygame.transform.scale(pygame.image.load(os.path.join('video', 'pics', 'tank_tower.png')), (round(tower_width), round(tower_height)))


tank_tower = pygame.Surface((round(tower_width*2), round(tower_height*2)))
tank_tower.blit(img_tank_tower, (round(tower_width)-img_tank_tower.get_width()//2, round(tower_height) - img_tank_tower.get_height()//2))



def play_video(game):
    game_end = False
    move = 0
    turn = 0
    tower = 0
    shot = False
    coll_map = pygame.transform.rotate(pygame.surfarray.make_surface(game.connection.env_from_server[0][:, :, :3] * 255), 0)
    coll_map = pygame.transform.scale(coll_map, (WIDTH * MULTY_PIXEL, HEIGHT * MULTY_PIXEL))
    # coll_map2 = pygame.transform.rotate(pygame.surfarray.make_surface(game.map_coll[:, :, :3] * 255), 0)
    # coll_map2 = pygame.transform.scale(coll_map2, (WIDTH * MULTY_PIXEL, HEIGHT * MULTY_PIXEL))

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

        # for hand control
        game.connection.send_action(0, [move, turn, tower, shot, False])

    DISPLAY.blit(coll_map, (DISPLAY.get_width() // 2, 0))
    # DISPLAY.blit(coll_map2, (0, 0))
    # DISPLAY.blit(features_nn, (0, DISPLAY.get_height() // 2))

    DISPLAY.blit(background, (0, 0))

    for tank in game.team1:
        # TODO check by hp, if == 0 than show dead tank
        tank_body = pygame.transform.rotate(img_tank_base, tank.direction_tank*360)
        # tank_rect = tank_body.get_rect(center=((tank_width//2), (tank_height//2)))
        DISPLAY.blit(tank_body, (tank.X * MULTY_PIXEL, tank.Y*MULTY_PIXEL))

        # angle = tank.direction_tank+tank.direction_tower
        # xx = 0.6*0.5 * np.cos(np.pi * 2 * angle) - 1*0.5 * np.sin(
        #     np.pi * 2 * angle)
        # yy = 0.6*0.5 * np.sin(np.pi * 2 * angle) + 1*0.5 * np.cos(
        #     np.pi * 2 * angle)
        # print(round(angle, 2), 'x:', round(xx, 1), 'y:', round(yy, 1))
        tower = pygame.transform.rotate(img_tank_tower, (tank.direction_tank+tank.direction_tower) * 360)
        tower_rect = tower.get_rect()
        DISPLAY.blit(tower, (tower_rect[0] + tank.X* MULTY_PIXEL+ tank.crop_x, tower_rect[1] + tank.Y * MULTY_PIXEL + tank.crop_y)) # (tank.X * MULTY_PIXEL + tower.get_width()//2, tank.Y * MULTY_PIXEL + tower.get_height()//2))

    for tank in game.team2:
        tank_body = pygame.transform.rotate(img_tank_base, tank.direction_tank * 360)
        tank_rect = tank_body.get_rect(center=(tank_width // 2, tank_height // 2))
        DISPLAY.blit(tank_body, (tank.X * MULTY_PIXEL + tank_rect[0], tank.Y * MULTY_PIXEL + tank_rect[1]))

    # Bullets
    for i in game.bullets_in_act:
        pygame.draw.circle(DISPLAY, (255,0,0), (game.bullets[i].X*MULTY_PIXEL,  game.bullets[i].Y*MULTY_PIXEL), 3)



    pygame.display.update()
    time.sleep(1 / FRAME_RATE)
