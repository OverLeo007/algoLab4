from random import shuffle
import pygame as pg
# import sys
# import argparse
# import os
from math import ceil
# from typing import Optional, Callable
import colour
from numba import njit
import numpy as np
import winsound

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(
            f'func: {f.__name__} len: {len(args[0])} '
            f'method: {kw["method"] if kw.get("method", False) else "auto"} '
            f'time: {te - ts:2.4f} sec')
        return result

    return wrap


def normalize_color(color):
    return tuple(col / 255 for col in color)


def col_to_bytes(col_array):
    return [tuple(map(lambda x: int(x * 255), color.get_rgb())) for color in col_array]


def get_grad(length, col1, col2):
    color1 = colour.Color(rgb=normalize_color(col1))
    color2 = colour.Color(rgb=normalize_color(col2))
    col_array = list(color1.range_to(color2, length))
    return np.array(col_to_bytes(col_array))


@njit(fastmath=True)
def another_cols_array(array, ar_len, col_array, width, height, red, res_array, red_num):
    col_count = ar_len
    norm_x = width / col_count
    max_el = max(array)
    for index, value in enumerate(array):
        norm_h = int((value / max_el) * (height - 100))
        norm_y = height - norm_h
        res = int(value / max_el * col_count)

        for x in range(int(norm_x * index), int(norm_x * (index + 1))):
            for y in range(int(norm_y), int(norm_y + norm_h)):
                res_array[x, y] = red_num if index == red else col_array[res - 1]

    return res_array


def draw_array_col(array: list, red, width: int, height: int,
                   screen: pg.Surface, colors_array, method="default"):
    array = np.array(array)

    screen.fill((0, 0, 0))
    if method == "default":
        col_count = len(array)
        max_el = max(array)
        norm_x = width / col_count
        norm_w = norm_x if norm_x > 1 else 1

        for index, value in enumerate(array):
            norm_h = (value / max_el) * (height - 100)
            norm_y = height - norm_h

            if index != red:
                cur_color = colors_array[int(value / max_el * col_count) - 1]
            else:
                cur_color = (255, 0, 0)
            pg.draw.rect(screen, cur_color, (ceil(norm_x * index), norm_y, ceil(norm_w), norm_h))
    else:
        columns = another_cols_array(array, len(array), colors_array, width, height, red,
                                     np.full((width, height) + (3,), [0, 0, 0]),
                                     np.array((255, 0, 0)))
        pg.surfarray.blit_array(screen, columns)

    pg.display.update()
    pg.time.wait(1)


def make_array(length):
    array = [i for i in range(1, length)]
    shuffle(array)
    return array


def draw_sort():
    pg.init()
    width, height = 800, 600
    screen = pg.display.set_mode((width, height))
    pg.display.set_caption("MergeSort visualize")
    ar_len = 50
    array = make_array(ar_len)
    colors_array = get_grad(ar_len, *((255, 0, 0), (0, 255, 0)))

    # frequency = 3000  # Set Frequency To 2500 Hertz
    # duration = 200  # Set Duration To 1000 ms == 1 second
    #
    #
    # exit()

    @timing
    def my_sort(array_to_sort: list, reverse: bool = False, method=None):
        if method is None:
            if len(array_to_sort) > 220:
                method = "optimized"
            else:
                method = "default"

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
                draw_array_col(arr, i, width, height, screen, colors_array, method)

        return merge_sort(array_to_sort, 0, len(array_to_sort) - 1)

    # my_sort(array)
    run = True
    while run:
        pg.display.update()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                run = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_r:
                    shuffle(array)
                if event.key == pg.K_UP:
                    shuffle(array)
                    my_sort(array, method="default")
                if event.key == pg.K_DOWN:
                    shuffle(array)
                    my_sort(array, method="optimized")


if __name__ == '__main__':
    draw_sort()
    # print(get_grad([i for i in range(0, 100)], (255, 0, 0), (0, 0, 255)))
    # print(my_sort([randint(1, 100) for _ in range(100)]))
    # print(color_constructor("0.0.5"))
