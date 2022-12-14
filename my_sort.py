from sys import getrecursionlimit

from typing import Optional, Callable


def my_sort(array: list, reverse: bool = False,
            key: Optional[Callable] = None,
            cmp: Optional[Callable] = None) -> list:
    """
    Реализация сортировки слиянием

    @param array: сортируемый массив
    @param reverse: сортируем напрямую или в обратную сторону
    @param key: ключ сортировки в виде функции
    @param cmp: компаратор для значений
    @return: отсортированный массив
    """
    array = list(array)
    key = key if key is not None else lambda x: x
    cmp = cmp if cmp is not None else lambda x, y: x < y

    def insertion_sort(arr):
        for i in range(1, len(arr)):
            temp = arr[i]
            j = i - 1
            while j >= 0 and \
                    (cmp(key(temp),  key(arr[j])) if reverse else cmp(key(arr[j]),  key(temp))):
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = temp

    def merge_sort(arr: list, depth: int = 1) -> list:

        if (n_len := len(arr)) > 1:
            mid = n_len // 2
            left = arr[:mid]
            right = arr[mid:]
            if depth + 1 > getrecursionlimit():
                insertion_sort(left)
                insertion_sort(right)
            else:
                merge_sort(left, depth + 1)
                merge_sort(right, depth + 1)

            i = j = k = 0

            while i < len(left) and j < len(right):
                if cmp(key(left[i]), key(right[j])) \
                        if not reverse else \
                        cmp(key(right[j]), key(left[i])):
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1

            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1

            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1
        return arr

    return merge_sort(array)


if __name__ == '__main__':
    from random import randint

    lst = [randint(1, 1000) for i in range(1000)]
    print(my_sort(lst))
