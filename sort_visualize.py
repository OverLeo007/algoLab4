import argparse
from random import shuffle
import pygame as pg
from math import ceil
from typing import Optional, Callable, Tuple, List
import colour
from numba import njit
import numpy as np
from my_sort import my_sort as ms

from functools import wraps
from time import time


def timing(f: Callable):
    """
    Декоратор для засечения времени выполнения отрисовки
    @param f: функция, время которой засекаем
    """
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


def normalize_color(color: tuple[int, int, int]) -> tuple[float, ...]:
    """
    Приведение цвета к нормальному виду

    @param color: цвет в формате (0-255, 0-255, 0-255)
    @return: цвет в формате (0-1, 0-1, 0-1)
    """
    return tuple(col / 255 for col in color)


def col_to_bytes(col_array: list[colour.Color, ...]) -> list[tuple[int, ...]]:
    """
    Приведение цвета из нормального вида к (255, 0, 0)
    @param col_array: (0-1, 0-1, 0-1)
    @return: цвет в формате (0-255, 0-255, 0-255)
    """
    return [tuple(map(lambda x: int(x * 255), color.get_rgb())) for color in col_array]


def get_grad(length: int, col1: tuple[int, int, int], col2: tuple[int, int, int]) -> np.ndarray:
    """
    Создает список градиентного перехода от col1 до col2
    @param length: длина списка
    @param col1: цвет начала градиента
    @param col2: цвет конца градиента
    @return: список градиентного перехода от col1 до col2
    """
    color1 = colour.Color(rgb=normalize_color(col1))
    color2 = colour.Color(rgb=normalize_color(col2))
    col_array = list(color1.range_to(color2, length))
    return np.array(col_to_bytes(col_array))


@njit(fastmath=True)
def another_cols_array(array, max_el, ar_len, col_array,
                       width, height, red, res_array, red_num):
    col_count = ar_len
    norm_x = width / col_count
    for index, value in enumerate(array):
        norm_h = int((value / max_el) * (height - 100))
        norm_y = height - norm_h
        res = int(value / max_el * col_count)

        for x in range(int(norm_x * index), int(norm_x * (index + 1))):
            for y in range(int(norm_y), int(norm_y + norm_h)):
                res_array[x, y] = red_num if index == red else col_array[res - 1]
    return res_array


def draw_array_col(array: np.ndarray, max_el: int, length: int,
                   red: int, width: int, height: int,
                   screen: pg.Surface, colors_array: np.ndarray, tick: int):
    """
    Отрисовка нового состояния массива

    @param array: массив
    @param max_el: максимальный элемент массива
    @param length: длина массива
    @param red: текущее положение изменяемого элемента
    @param width: ширина экрана
    @param height: длина экрана
    @param screen: surface на котором рисуем
    @param colors_array: список градиентного перехода, определенный для всего массива
    @param tick: сколько времени должна занимать отрисовка
    """
    norm_x = width / length
    norm_w = norm_x if norm_x > 1 else 1
    if red == 0:
        screen.fill((0, 0, 0))
    else:
        screen.fill((0, 0, 0), rect=(0, 0, (red + 1) * norm_x, height))
    h_caf = (height - 100) / max_el

    for index, value in enumerate(array):

        norm_h = value * h_caf
        norm_y = height - norm_h

        if index != red or value == max_el:
            cur_color = colors_array[int(value / max_el * length) - 1]
        else:
            cur_color = (255, 0, 0)
        if red != 0 and index == red + 1:
            break
        pg.draw.rect(screen, cur_color, (ceil(norm_x * index), norm_y, ceil(norm_w), norm_h))

    # screen.fill((0, 0, 0))
    # columns = another_cols_array(array, max_el, length, colors_array, width, height, red,
    #                              np.full((width, height) + (3,), [0, 0, 0]),
    #                              np.array((255, 0, 0)))
    # pg.surfarray.blit_array(screen, columns)

    pg.display.update()
    pg.time.wait(tick)


def draw_sort(array: np.ndarray, reverse: bool = False,
              colors: tuple[tuple[int, int, int], tuple[int, int, int]] =
              ((255, 255, 255), (255, 255, 255))):
    """
    Функция отрисовки процесса сортировки
    @param array: сортируемый массив
    @param reverse: нужно ли сортировать по неубыванию
    @param colors: цвета для создания градиентного перехода от colors[0] до colors[1]
    """
    pg.init()
    width, height = 800, 600
    screen = pg.display.set_mode((width, height))
    pg.display.set_caption("MergeSort visualize")

    ar_len = len(array)
    if ar_len <= 100:
        tick = 20
    else:
        tick = 1
    colors_array = get_grad(ar_len, *colors)
    max_element = max(array)

    @timing
    def my_sort(array_to_sort: np.ndarray):
        """
        Функция сортировки

        @param array_to_sort: сортируемый массив
        """
        def merge_sort(arr: np.ndarray, left, right):
            """
            Основная функция сортировки слиянием, распределяющая границы сортируемых частей
            @param arr: исходный массив
            @param left: индекс левого элемента
            @param right: индекс правого элемента
            @return: отсортированный массив
            """
            mid = (left + right) // 2
            if left < right:
                merge_sort(arr, left, mid)
                merge_sort(arr, mid + 1, right)
                merge(arr, left, mid, mid + 1, right)
            return arr

        def merge(arr: np.ndarray, left1, right1, left2, right2):
            """
            Функция слияния двух частей
            @param arr: исходный массив
            @param left1: левая граница левой части
            @param right1: правая граница левой части
            @param left2: левая граница правой части
            @param right2: правая граница правой части
            """
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
                draw_array_col(arr, max_element, ar_len, i, width, height, screen, colors_array,
                               tick)

        return merge_sort(array_to_sort, 0, len(array_to_sort) - 1)

    my_sort(array)
    run = True
    while run:
        pg.display.update()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                run = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    shuffle(array)
                    my_sort(array)


def main():
    """
    Точка входа из для CLI
    """
    parser = argparse.ArgumentParser(description="Сортировка методом слияния")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--array", "-a", dest="array", type=int, nargs="+",
                       help="Список чисел через пробел")
    group.add_argument("--file_path", "-fp", dest="file_path",
                       help="Путь к файлу с числами расположенными через пробел")
    group.add_argument("--randomized_array", "-ra", dest="ra_len", type=int,
                       help="Длина для создания рандомного массива")
    parser.add_argument("--reverse", "-r", dest="reverse",
                        type=argparse.BooleanOptionalAction,
                        help="Если указано - сортирует по невозрастанию")
    parser.add_argument("--visualize", "-v", dest="visualize",
                        action=argparse.BooleanOptionalAction,
                        help="visualize array while sorting")
    parser.add_argument("--colors", "-c", type=int, nargs=6,
                        help="Цвета наименьшего и наибольшего элемента массива, "
                             "для создания градиента")
    args = parser.parse_args()

    res = {"array": None,
           "reverse": False}

    if args.array:
        res["array"] = np.array(args.array)
    elif args.file_path:
        with open(args.file_path, "r") as file:
            res["array"] = np.array(list(map(int, file.read().replace("\n", "").split(" "))))
    elif args.ra_len:
        tmp = [i for i in range(1, args.ra_len)]
        shuffle(tmp)
        res["array"] = np.array(tmp)

    if args.reverse:
        res["reverse"] = args.reverse

    if args.visualize:
        if args.colors:
            r1, g1, b1, r2, g2, b2 = args.colors
            res["colors"] = ((r1, g1, b1), (r2, g2, b2))
        draw_sort(**res)
    else:
        print(ms(**res))


if __name__ == '__main__':
    arr_to_s = np.array([i for i in range(1, 500)])
    shuffle(arr_to_s)
    draw_sort(arr_to_s, colors=((255, 0, 0), (0, 0, 255)))
    # main()
