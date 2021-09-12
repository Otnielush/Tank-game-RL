import os
import time

import numpy as np
import pygame
from options.video import WIDTH, HEIGHT, FRAME_RATE

MULTY_PIXEL_V = 50

# run display
# pygame.init()
clock = pygame.time.Clock()

def init_display(WIDTH, HEIGHT):
    global DISPLAY, pygame, tank_obs, tank_obs2, tank_obs3, flag_red, flag_blue, tank_destroyed, land, rock, wall, \
        forest, water, desert, swamp, bush, background, img_tank_base_red, img_tank_base_blue, tank_tower, \
        img_tank_tower, tank_width, tank_height
    DISPLAY = pygame.display.set_mode((WIDTH * MULTY_PIXEL_V, HEIGHT * MULTY_PIXEL_V))
    pygame.init()
    pygame.display.update()
    pygame.display.set_caption("Tank game RL")


    # loading pics
    # obstacles
    tank_obs = pygame.image.load(os.path.join('video', 'pics', 'tank_obs.webp'))
    tank_obs2 = pygame.image.load(os.path.join('video', 'pics', 'retro-pixel.webp'))
    tank_obs3 = pygame.image.load(os.path.join('video', 'pics', 'Tank Town V2.png'))
    flag_pic = pygame.image.load(os.path.join('video', 'pics', 'flag.png'))

    flag_red = pygame.transform.scale(flag_pic, (MULTY_PIXEL_V, MULTY_PIXEL_V))
    flag_red.fill((255, 0, 0, 100), special_flags=pygame.BLEND_ADD)
    flag_blue = pygame.transform.scale(flag_pic, (MULTY_PIXEL_V, MULTY_PIXEL_V))
    flag_blue.fill((0, 0, 255, 100), special_flags=pygame.BLEND_ADD)

    tank_destroyed = tank_obs3.subsurface((113, 112, 15, 15))
    temp = tank_obs3.subsurface((144, 48, 15, 15))
    land = pygame.transform.scale(temp, (MULTY_PIXEL_V, MULTY_PIXEL_V))
    land.fill((139, 69, 19, 100), special_flags=pygame.BLEND_ADD)
    temp = tank_obs.subsurface((116, 15, 28, 29))
    rock = pygame.transform.scale(temp, (MULTY_PIXEL_V, MULTY_PIXEL_V))
    temp = tank_obs.subsurface((30, 159, 28, 29))
    wall = pygame.transform.scale(temp, (MULTY_PIXEL_V, MULTY_PIXEL_V))
    temp = tank_obs.subsurface((58, 15, 28, 29))
    forest = pygame.transform.scale(temp, (MULTY_PIXEL_V, MULTY_PIXEL_V)).convert_alpha()
    # all colors changing to black, except green
    pygame.transform.threshold(
        dest_surf=forest,
        surf=forest.copy(),
        search_color=(8, 63, 33),
        threshold=(50, 50, 50),
        set_color=(0, 0, 0),
        set_behavior=1,
        search_surf=None,
        inverse_set=True
    )
    forest.set_colorkey((0, 0, 0))
    temp = tank_obs.subsurface((116, 101, 28, 29))
    water = pygame.transform.scale(temp, (MULTY_PIXEL_V, MULTY_PIXEL_V))
    temp = tank_obs.subsurface((87, 362, 28, 29))
    desert = pygame.transform.scale(temp, (MULTY_PIXEL_V, MULTY_PIXEL_V))
    desert.fill((240, 240, 0, 100), special_flags=pygame.BLEND_MAX)
    temp = tank_obs2.subsurface((192, 347, 32, 30))
    temp2 = pygame.transform.scale(temp, (MULTY_PIXEL_V // 3, MULTY_PIXEL_V // 3))
    bush = pygame.Surface((MULTY_PIXEL_V, MULTY_PIXEL_V))
    for x in range(2):
        for y in range(2):
            bush.blit(temp2,
                      (MULTY_PIXEL_V // 8 + MULTY_PIXEL_V // 4 * x * 2, MULTY_PIXEL_V // 8 + MULTY_PIXEL_V // 4 * y * 2))
    # black is now transparent
    bush.set_colorkey((0, 0, 0))
    temp = tank_obs.subsurface((87, 362, 28, 29))
    desert = pygame.transform.scale(temp, (MULTY_PIXEL_V, MULTY_PIXEL_V))
    temp = tank_obs3.subsurface((112, 48, 15, 15))
    swamp = pygame.transform.scale(temp, (MULTY_PIXEL_V, MULTY_PIXEL_V))
    swamp.fill((139, 69, 19, 100), special_flags=pygame.BLEND_ADD)
    # {'land': 0, 'bush': 0.14, 'desert': 0.29, 'forest': 0.43, 'water': 0.57, 'swamp': 0.71, 'wall': 0.86, 'rock': 1}

    # background
    background = pygame.surface.Surface((WIDTH * MULTY_PIXEL_V, HEIGHT * MULTY_PIXEL_V))
    background.fill((139, 69, 19))
    color_lines = (250, 250, 100)
    for x in np.arange(1, WIDTH):
        pygame.draw.lines(background, color_lines, False,
                          [(x * MULTY_PIXEL_V, 0), (x * MULTY_PIXEL_V, HEIGHT * MULTY_PIXEL_V)], 1)
    for y in np.arange(1, HEIGHT):
        pygame.draw.lines(background, color_lines, False,
                          [(0, y * MULTY_PIXEL_V), (WIDTH * MULTY_PIXEL_V, y * MULTY_PIXEL_V)], 1)



    # colors

    # TODO change to scaling by type tank
    tower_width = 0.4 * MULTY_PIXEL_V
    tower_height = 1 * MULTY_PIXEL_V
    tank_width = round(0.6 * MULTY_PIXEL_V)
    tank_height = 1 * MULTY_PIXEL_V
    # img_tank_base = pygame.transform.scale(pygame.image.load(os.path.join('video', 'pics', 'tank.png')), (tank_width, tank_height))
    # img_tank_tower = pygame.transform.scale(pygame.image.load(os.path.join('video', 'pics', 'tank_tower.png')), (round(tower_width), round(tower_height)))
    img_tank_base_red = pygame.transform.scale(pygame.image.load(os.path.join('video', 'pics', 'tankBase2.png')),
                                           (tank_width, tank_height))
    img_tank_base_red.fill((100, 0, 0, 1), special_flags=pygame.BLEND_ADD)
    img_tank_base_blue = pygame.transform.scale(pygame.image.load(os.path.join('video', 'pics', 'tankBase2.png')),
                                           (tank_width, tank_height))
    img_tank_base_blue.fill((0, 0, 100, 1), special_flags=pygame.BLEND_ADD)

    img_tank_tower = pygame.transform.scale(pygame.image.load(os.path.join('video', 'pics', 'tankTurret.png')),
                                            (round(tower_width), round(tower_height)))

    tank_tower = pygame.Surface((round(tower_width * 2), round(tower_height * 2)))
    tank_tower.blit(img_tank_tower, (
    round(tower_width) - img_tank_tower.get_width() // 2, round(tower_height) - img_tank_tower.get_height() // 2))

    del(temp, temp2)


# for backgroud building
def video_build_map(game):
    global background, pygame

    background.fill((139, 69, 19))
    # drawing background from game map
    for x in range(game.width):
        for y in range(game.height):
            # making background picture from game map
            idd = game.map_env[x, y, 0]
            idd_base_red = game.map_env[x, y, 1]
            idd_base_blue = game.map_env[x, y, 2]

            if idd == 0:
                background.blit(land, (x * MULTY_PIXEL_V, y * MULTY_PIXEL_V))
            elif idd == 1:
                background.blit(rock, (x * MULTY_PIXEL_V, y * MULTY_PIXEL_V))
            elif idd == 0.86:
                background.blit(wall, (x * MULTY_PIXEL_V, y * MULTY_PIXEL_V))
            elif idd == 0.43:
                background.blit(forest, (x * MULTY_PIXEL_V, y * MULTY_PIXEL_V))
            elif idd == 0.57:
                background.blit(water, (x * MULTY_PIXEL_V, y * MULTY_PIXEL_V))
            elif idd == 0.29:
                background.blit(desert, (x * MULTY_PIXEL_V, y * MULTY_PIXEL_V))
            elif idd == 0.14:
                background.blit(bush, (x * MULTY_PIXEL_V, y * MULTY_PIXEL_V))
            elif idd == 0.71:
                background.blit(swamp, (x * MULTY_PIXEL_V, y * MULTY_PIXEL_V))

            if idd_base_red == 1:
                background.blit(flag_red, (x * MULTY_PIXEL_V, y * MULTY_PIXEL_V))
            elif idd_base_blue == 1:
                background.blit(flag_blue, (x * MULTY_PIXEL_V, y * MULTY_PIXEL_V))


# TODO play explosions from ammoes and tanks
def play_video(game, VIDEO):
    move = 0
    turn = 0
    tower = 0
    shot = False
    # coll_map = pygame.transform.rotate(pygame.surfarray.make_surface(game.connection.env_from_server[0][:, :, :3] * 255), 0)
    # coll_map = pygame.transform.scale(coll_map, (WIDTH * MULTY_PIXEL_V, HEIGHT * MULTY_PIXEL_V))
    # coll_map = pygame.transform.rotate(pygame.surfarray.make_surface(game.map_coll[:, :, :3] * 255), 0)
    # coll_map = pygame.transform.scale(coll_map, (WIDTH * MULTY_PIXEL_V, HEIGHT * MULTY_PIXEL_V))

    # features_nn = pygame.transform.rotate(pygame.surfarray.make_surface(game.map_env[:, :, :3] * 255), 0)
    # features_nn = pygame.transform.scale(features_nn, (WIDTH * MULTY_PIXEL_V, HEIGHT * MULTY_PIXEL_V))

    for event in pygame.event.get():
        # closing window but not game
        if event.type == pygame.QUIT:
            VIDEO[0] = False
            pygame.quit()
            return

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
        game.connection.send_action(103, [move, turn, tower, shot, False])

    # DISPLAY.blit(coll_map, (DISPLAY.get_width() // 2, 0))
    # DISPLAY.blit(features_nn, (0, DISPLAY.get_height() // 2))

    DISPLAY.blit(background, (0, 0))

    for tank in game.team1:
        if tank.hp <= 0:
            continue
        # TODO check by hp, if == 0 than show dead tank
        tank_body = pygame.transform.rotate(img_tank_base_red, tank.direction_tank * 360)
        # tank_rect = tank_body.get_rect(center=((tank_width//2), (tank_height//2)))
        DISPLAY.blit(tank_body, (tank.X * MULTY_PIXEL_V, tank.Y * MULTY_PIXEL_V))

        # angle = tank.direction_tank+tank.direction_tower
        # xx = 0.6*0.5 * np.cos(np.pi * 2 * angle) - 1*0.5 * np.sin(
        #     np.pi * 2 * angle)
        # yy = 0.6*0.5 * np.sin(np.pi * 2 * angle) + 1*0.5 * np.cos(
        #     np.pi * 2 * angle)
        # print(round(angle, 2), 'x:', round(xx, 1), 'y:', round(yy, 1))
        tower = pygame.transform.rotate(img_tank_tower, (tank.direction_tank + tank.direction_tower) * 360)
        tower_rect = tower.get_rect()
        DISPLAY.blit(tower, (tower_rect[0] + tank.X * MULTY_PIXEL_V, tower_rect[
            1] + tank.Y * MULTY_PIXEL_V))  # (tank.X * MULTY_PIXEL_V + tower.get_width()//2, tank.Y * MULTY_PIXEL_V + tower.get_height()//2))

    for tank in game.team2:
        if tank.hp <= 0:
            continue
        tank_body = pygame.transform.rotate(img_tank_base_blue, tank.direction_tank * 360)
        # tank_rect = tank_body.get_rect(center=(tank_width // 2, tank_height // 2))
        DISPLAY.blit(tank_body, (tank.X * MULTY_PIXEL_V, tank.Y * MULTY_PIXEL_V))
        tower = pygame.transform.rotate(img_tank_tower, (tank.direction_tank + tank.direction_tower) * 360)
        tower_rect = tower.get_rect()
        DISPLAY.blit(tower, (tower_rect[0] + tank.X * MULTY_PIXEL_V + tank.crop_x, tower_rect[
            1] + tank.Y * MULTY_PIXEL_V + tank.crop_y))  # (tank.X * MULTY_PIXEL_V + tower.get_width()//2, tank.Y * MULTY_PIXEL_V + tower.get_height()//2))

    # Bullets
    for i in game.bullets_in_act:
        if game.bullets[i].type == 'laser':
            pygame.draw.line(DISPLAY, (25, 255, 25), [int(game.bullets[i].X*MULTY_PIXEL_V), int(game.bullets[i].Y*MULTY_PIXEL_V)],
                             [int(game.bullets[i].end_xy[0]*MULTY_PIXEL_V), int(game.bullets[i].end_xy[1]*MULTY_PIXEL_V)], 2)
        else:
            pygame.draw.circle(DISPLAY, (255, 0, 0), (int(game.bullets[i].X * MULTY_PIXEL_V), int(game.bullets[i].Y * MULTY_PIXEL_V)), 3)

    pygame.display.update()
    time.sleep(1 / (FRAME_RATE + 15))


def destroy_tank(x, y, width, height, direction):
    global background
    tank_body = pygame.transform.rotate(pygame.transform.scale(tank_destroyed,
                (int(round(width * MULTY_PIXEL_V)), int(round(height * MULTY_PIXEL_V)))), direction * 360 + 180)
    background.blit(tank_body, (x * MULTY_PIXEL_V, y * MULTY_PIXEL_V))

def destroy_wall(x, y):
    global background
    background.blit(land, (int(x) * MULTY_PIXEL_V, int(y) * MULTY_PIXEL_V))

def close_window():
    if pygame.display.get_init:
        pygame.quit()
