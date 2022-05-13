#!/usr/bin/env python3
"""
Khalikov Ruslan. 13.05.2022
На плоскости задана область.
Внутри области расположено множество точек с координатами, их N.
Также задана кусочно-линейная кривая  в виде упорядоченного набора узлов, их M.
Из множества  нужно выделить упорядоченное подмножество точек, которые расположены близко к кривой.

Формат входных данных:
N M
x1 y1
...
...
xN yN
индекс_1_точки_кривой ... индекс_М_точки_кривой

python3 task.py --file_in "input file" --file_out "out file" --min_dist "minimal dist"

"""
import sys
from argparse import ArgumentParser
import math
from typing import List, Tuple, Set

class Point:
    """
    класс точка
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Vector:
    """
    класс вектор
    """
    def __init__(self, point1, point2):
        self.x = point2.x - point1.x
        self.y = point2.y - point1.y

    def dot_product(self, vector2):
        """
        скалярное произведение
        :param vector2:
        :return:
        """
        return self.x * vector2.x + self.y * vector2.y

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--file_in", default="./test_01.txt")
    parser.add_argument("--file_out", default="./out_01.txt")
    parser.add_argument("--min_dist", default=5)
    return parser.parse_args()

def read_data(file_in):
    """
    :return:
    list_idx_line - индексы точек заданной кривой
    """
    with open(file_in, "r") as fin:
        N, M = map(int, fin.readline().split())
        P = []
        for _ in range(N):
            x, y = map(float, fin.readline().split())
            P.append(Point(x, y))

        list_idx_line = list(
            map(
                lambda x: x - 1,
                list(map(int, fin.readline().split()))
            )
        )
    C = [P[i] for i in list_idx_line]

    return N, M, P, C, list_idx_line

def calc_dist(ab, bc, ac, a, b, c) -> float:
    dot_prod_ab_bc = ab.dot_product(bc)
    dot_prod_ab_ac = ab.dot_product(ac)

    if dot_prod_ab_bc > 0:
        h_x = c.x - b.x
        h_y = c.y - b.y
        d = math.sqrt(h_x**2 + h_y**2)
    elif dot_prod_ab_ac < 0:
        h_x = c.x - a.x
        h_y = c.y - a.y
        d = math.sqrt(h_x**2 + h_y**2)
    else:
        x1 = ab.x
        x2 = ac.x
        y1 = ab.y
        y2 = ac.y
        d = abs(x1 * y2 - y1 * x2) / (math.sqrt(x1**2 + y1**2))

    return d

def get_dist(
        P, list_idx_not_in_line, idx, idx_1, min_dist, idx_answer: Set
) -> List[Tuple[Point, float]]:
    """
    Вычисляем расстояние от заданного сегмента(A, B), конечные точки которого
    находятся в точках с индексами idx(A), idx_1(B), до всех остальных точек(C)
    :param P:
    :param list_idx_not_in_line:
    :param idx:
    :param idx_1:
    :param idx_answer:
    :param min_dist:
    :return:
    """
    answer = []
    point_a = P[idx]
    point_b = P[idx_1]
    ab = Vector(point_a, point_b)
    for i in list_idx_not_in_line:
        point_c = P[i]
        bc = Vector(point_b, point_c)
        ac = Vector(point_a, point_c)
        dist = calc_dist(ab, bc, ac, point_a, point_b, point_c)
        if (dist <= min_dist and (i not in idx_answer)):
            answer.append((P[i], dist))
            idx_answer.add(i)
    return answer

def calculate(N, M, P, C, list_idx_line, args) -> Tuple[List[Tuple[Point, float]], Set]:
    """
    Находим расстояния от заданных сегментов до всех остальных точек
    :param N:
    :param M:
    :param P:
    :param C:
    :param list_idx_line:
    :param args:
    :return:
    """
    set_idx_line = set(list_idx_line)

    # список индексов точек которые не задают кривую
    list_idx_not_in_line = sorted(list(set(range(N)) - set_idx_line))

    temp_tilda = []
    idx_answer = set()
    min_dist = float(args.min_dist)
    for idx, idx_1 in zip(list_idx_line, list_idx_line[1:]):
        list_dist = get_dist(P, list_idx_not_in_line, idx, idx_1, min_dist, idx_answer)
        temp_tilda += list_dist

    return temp_tilda, idx_answer

def print_answer(set_tilda: Tuple[List[Tuple[Point, float]], Set], args):
    """
    вывод результата
    :param set_tilda:
    :param args:
    :return:
    """
    with open(args.file_out, "w") as fout:
        for (point, dist), idx in zip(set_tilda[0], set_tilda[1]):
            fout.write(f"point #{idx + 1}: x = {point.x}, y = {point.y}, dist = {dist}\n")

    return 0

def main(args):
    """
    здесь логика
    :param args:
    :return:
    """
    N, M, P, C, list_idx_line = read_data(args.file_in)
    set_tilda = calculate(N, M, P, C, list_idx_line, args)

    print_answer(set_tilda, args)


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
