from typing import Union
import numpy as np
from numlib.util import error as error


def sum(x : list[list[Union[int, float]]]) -> list[list[Union[int, float]]]:
    if isinstance(x, list) and isinstance(x[0], list):
        r = 0
        for i in range(len(x)):
            for j in range(len(x[i])):
                r += x[i][j]
        return r
    if isinstance(x, list):
        r = 0
        for i in range(len(x)):
            r += x[i]
        return r


def square(x : list[list[Union[int, float]]]) -> list[list[Union[int, float]]]:
    if isinstance(x , list) and isinstance(x[0], list):
        result = []
        for i in range(len(x)):
            row = []
            for j in range(len(x[i])):
                A = x[i][j] * x[i][j]
                row.append(A)
            result.append(row)
        return result
    if isinstance(x, list):
        result = []
        for i in range(len(x)):
            A = x[i] * x[i]
            result.append(A)
        return result
