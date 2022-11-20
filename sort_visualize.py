from random import randint, shuffle
import pygame as pg
import sys
import argparse
import os
import colour
import math
from numba import njit
import numpy as np

from typing import Optional, Callable


def normalize_color(color):
    return tuple(col / 255 for col in color)


def col_to_bytes(col_array):
    return [tuple(map(lambda x: int(x * 255), color.get_rgb())) for color in col_array]


def get_grad(val_array, col1, col2):
    color1 = colour.Color(rgb=normalize_color(col1))
    color2 = colour.Color(rgb=normalize_color(col2))
    col_array = list(color1.range_to(color2, len(val_array)))
    res = np.array(col_to_bytes(col_array))
    return res


# @njit(fastmath=True)
def get_cols_array(array, col_array, width, height, red):
    col_count = len(array)
    norm_x = width / col_count
    res_array = []
    for index, value in enumerate(array):
        if norm_x <= 1:
            norm_x = 1

        norm_h = int((value / max(array)) * (height - 100))
        norm_y = height - norm_h
        res = int(value / max(array) * col_count)
        if index != red:
            cur_color = col_array[res - 1]
        else:
            cur_color = (255, 0, 0)
        res_array.append(tuple((cur_color, (norm_x * index, norm_y, norm_x, norm_h))))
    return res_array


@njit(fastmath=True)
def another_cols_array(array, ar_len, col_array, width, height, red, res_array, red_num):
    col_count = ar_len
    norm_x = width / col_count
    for index, value in enumerate(array):
        # if norm_x <= 1:
        #     norm_x = 1

        norm_h = int((value / max(array)) * (height - 100))
        norm_y = height - norm_h
        res = int(value / max(array) * col_count)
        if index != red:
            cur_color = col_array[res - 1]
        else:
            cur_color = red_num
        for x in range(int(norm_x * index), int(norm_x * (index + 1))):
            for y in range(int(norm_y), int(norm_y + norm_h)):
                res_array[x, y] = cur_color

    return res_array


def draw_array_col(array: list, red, width: int, height: int,
                   screen: pg.Surface, tick: int, colors):
    array = np.array(array)
    colors_array = get_grad(array, *colors)

    screen.fill((0, 0, 0))
    columns = another_cols_array(array, len(array), colors_array, width, height, red, np.full((width, height) + (3,), [0, 0, 0]), np.array((255, 0, 0)))
    pg.surfarray.blit_array(screen, columns)
    # columns = get_cols_array(array, colors_array, width, height, red)
    # for col in columns:
    #     pg.draw.rect(screen, *col)
    # print(columns)
    # exit()

    pg.display.update()
    # pg.time.wait(tick)


def draw_sort():
    pg.init()
    width, height = 800, 600
    screen = pg.display.set_mode((width, height))
    pg.display.set_caption("MergeSort visualize")
    colors = ((255, 0, 0), (0, 255, 255))
    array = [i for i in range(1, 300)]
    shuffle(array)

    def my_sort(array: list, reverse: bool = False):

        def merge_sort(arr: list, left, right):
            mid = (left + right) // 2
            if left < right:
                merge_sort(arr, left, mid)
                merge_sort(arr, mid + 1, right)
                merge(arr, left, mid, mid + 1, right)
            return arr

        def merge(arr: list, left1, right1, left2, right2):
            i = left1
            j = left2
            temp = [0] * (right2 + 1 - left1)
            k = 0
            while i <= right1 and j <= right2:

                if arr[i] < arr[j] if not reverse else arr[i] > arr[j]:

                    temp[k] = arr[i]
                    i += 1
                    k += 1
                else:
                    temp[k] = arr[j]
                    j += 1
                    k += 1

            while i <= right1:
                temp[k] = arr[i]
                i += 1
                k += 1

            while j <= right2:
                temp[k] = arr[j]
                j += 1
                k += 1

            j = 0
            for i in range(left1, right2 + 1):
                arr[i] = temp[j]
                j += 1
                draw_array_col(arr, i, width, height, screen, 1, colors)

        return merge_sort(array, 0, len(array) - 1)

    my_sort(array)
    run = True
    while run:
        pg.display.update()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                run = False
            if event.type == pg.KEYDOWN:
                pass
                # if event.key == pg.K_UP:
                # if event.type == pg.K_DOWN:
                #     my_sort(sorted(array))


def main():
    pg.init()
    width, height = 800, 600
    screen = pg.display.set_mode((width, height))
    pg.display.set_caption("MergeSort visualize")
    max_val = 100
    array = [randint(1, max_val) for _ in range(1, 1000)]
    # array = [i for i in range(1000)]
    # color1 = colour.Color(rgb=(255, 0, 0))
    # color2 = colour.Color(rgb=(0, 255, 255))
    # color_array = list(color1.range_to(color2, len(array)))
    # print(color_array)
    # draw_array_col(array, width, height, screen, 1)

    run = True
    while run:
        pg.display.update()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                run = False
            # if event.type == pg.KEYDOWN:
            #     if event.key == pg.K_UP:
            #         my_sort(args_array, reverse)


if __name__ == '__main__':
    draw_sort()
    # print(get_grad([i for i in range(0, 100)], (255, 0, 0), (0, 0, 255)))
    # print(my_sort([randint(1, 100) for _ in range(100)]))
    # print(color_constructor("0.0.5"))
