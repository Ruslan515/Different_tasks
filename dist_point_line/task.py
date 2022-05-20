#!/usr/bin/env python3
"""
Khalikov Ruslan. 20.05.2022
На плоскости задана область.
Внутри области расположено множество точек с координатами, их N.
Также задана кусочно-линейная кривая  в виде упорядоченного набора узлов, их M.
Из множества  нужно выделить упорядоченное подмножество точек, которые расположены близко к кривой.

Формат входных данных:
N M
индекс_1_точки_кривой ... индекс_М_точки_кривой

python3 task.py --file_in "input file" --file_out "out file" --image_out "out_image.png" --min_dist "minimal dist"

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
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

def read_data(file_in):
    """
    :return:
    list_idx_line - индексы точек заданной кривой
    """
    with open(file_in, "r") as fin:
        N, M = map(int, fin.readline().split())
        P = []
        # for _ in range(N):
        #     x, y = map(float, fin.readline().split())
        #     P.append(Point(x, y))

        x, y = list(np.random.uniform(-10, 10, size=N)), list(np.random.uniform(-10, 10, size=N))
        P = [Point(x_i, y_i) for x_i, y_i in zip(x, y)]

        list_idx_line = list(
            map(
                lambda x: x - 1,
                list(map(int, fin.readline().split()))
            )
        )
    C = [P[i] for i in list_idx_line]

    return N, M, P, C, list_idx_line

def euclid_dist(first: Point, second: Point) -> float:
    """
    евклидово расстояние между точками
    :param first:
    :param second:
    :return:
    """
    h_x = first.x - second.x
    h_y = first.y - second.y
    e_dist = math.sqrt(h_x**2 + h_y**2)
    return e_dist

def calc_dist(ab, bc, ac, a, b, c) -> Tuple[float, float]:
    """
    Вычисляет расстояние от заданной точки до семгента.
    так же до первой точки сегмента - А
    :param ab:
    :param bc:
    :param ac:
    :param a:
    :param b:
    :param c:
    :return:
    """
    dot_prod_ab_bc = ab.dot_product(bc)
    dot_prod_ab_ac = ab.dot_product(ac)

    dist_to_first_point = euclid_dist(c, a)

    if dot_prod_ab_bc > 0:
        dist_to_segment = euclid_dist(c, b)
    elif dot_prod_ab_ac < 0:
        dist_to_segment= dist_to_first_point
    else:
        x1 = ab.x
        x2 = ac.x
        y1 = ab.y
        y2 = ac.y
        dist_to_segment= abs(x1 * y2 - y1 * x2) / (math.sqrt(x1**2 + y1**2))

    return dist_to_segment, dist_to_first_point

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
    answer = {}
    """
    answer - словарь вида:
    ключ - пара(точка, расстояние до сегмента)
    значение - расстояние до первой точки сегмента.    
    """
    point_a = P[idx]
    point_b = P[idx_1]
    ab = Vector(point_a, point_b)
    for i in list_idx_not_in_line:
        point_c = P[i]
        bc = Vector(point_b, point_c)
        ac = Vector(point_a, point_c)
        dist_to_segment, dist_to_first_point = calc_dist(ab, bc, ac, point_a, point_b, point_c)
        if (dist_to_segment <= min_dist and (i not in idx_answer)):
            answer[(P[i], dist_to_segment)] = dist_to_first_point
            idx_answer.add(i)

    """
    получили словарь. теперь отсортируем его по расстоянию до первой точки сегмента    
    """
    answer = [x[0] for x in sorted(answer.items(), key=lambda x: x[1])]
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

def draw_dotes(P: List[Point], C: List[Point], answer_dots: List[Point], image_out) -> int:
    """
    нарисуем наши точки, линию и выходные точки
    :param P:
    :param C:
    :param answer_dots:
    :param image_out:
    :return:
    """
    x, y = [point.x for point in P], [point.y for point in P]
    plt.plot(x, y, "o")

    # for point in P:
    #     plt.annotate(f"({point.x:0.2f};{point.y:0.2f})", (point.x, point.y))

    x, y = [point.x for point in answer_dots], [point.y for point in answer_dots]
    plt.plot(x, y, "x", color='red')
    for i, point in enumerate(answer_dots):
        plt.annotate(f"{i} - ({point.x:0.2f};{point.y:0.2f})", (point.x, point.y))


    x, y = [point.x for point in C], [point.y for point in C]
    plt.plot(x, y)
    plt.savefig(image_out)

    return 0

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
    draw_dotes(P, C, [p for p, d in set_tilda[0]], args.image_out)

    print_answer(set_tilda, args)

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--file_in", default="./test.txt")
    parser.add_argument("--file_out", default="./out.txt")
    parser.add_argument("--image_out", default="./out.png")
    parser.add_argument("--min_dist", default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
