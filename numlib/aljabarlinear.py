import numpy as np
import math
from typing import Union
from util import error as error


def identity(n : int = 2) -> list[list[Union[int, float]]]:
    """
    Fungsi untuk membuat matriks identitas

    Parameter:
    n = panjang matriks

    return:
    Matriks Identitas
    """ 
    matriks = []
    for i in range(n):
        row = []
        for j in range(n):
            if (i == j):
                row.append(1)
            else:
                row.append(0)
        matriks.append(row)
    return matriks


def zeros(n : int , m : int = None) -> list[list[Union[int, float]]]:
    """
    Fungsi untuk membuat matriks nol

    parameter:
    n : ukuran matris n x n

    return:
    matriks : matriks nol
    """
    matriks = []
    if m == None:
        for _ in range(n):
            row = []
            for _ in range(n):
                row.append(0)
            matriks.append(row)
        return matriks
    else:
        for _ in range(n):
            row = []
            for _ in range(m):
                row.append(0)
            matriks.append(row)
        return matriks


def dot(A : Union[int, float], B : Union[int, float]) -> Union[int, float]:
    """
    Fungsi untuk melakukan perkalian dot

    Parameter :
    A : list
    B : list

    return :
    res : hasil perkalian dot
    """
    res = 0
    for i in range(len(A)):
        res += A[i] * B[i]
    return res


def cross(A : Union[int, float],
          B : Union[int, float]) -> Union[int, float]:
    """
    Fungsi untuk operasi cross product
    
    Parameter :
    A : matriks non-singular n * n
    B : matriks non-singular n * n

    Return :
    return hasil kalkulasi
    """
    if len(A) == len(B) == 3:
        return [
            A[1]*B[2] - A[2]*B[1],
            A[2]*B[0] - A[0]*B[2],
            A[0]*B[1] - A[1]*B[0]
        ]
    elif len(A) == len(B) == 2:
        return [
            A[1]*B[2] - A[2]*A[1]
        ]
    else:
        return error.IndeksError("panjang tidak sesuai")
    

def gauss_back_elim(
        A : list[list[Union[int, float]]],
        b : list[list[Union[int, float]]]
        ) -> list[Union[int, float]]:
    """
    Fungsi untuk melakukan eliminasi gauss

    parameter:
    A : Matriks non singular
    b : Matriks ruas kanan

    return:
    x : hasil kalkulasi fungsi
    """
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    for k in range(n - 1):
        if A[k, k] == 0:
            for i in range(k + 1, n):
                if A[i, k] != 0:
                    A[[k, i]] = A[[i, k]]
                    b[[k, i]] = b[[i, k]]
                    break
        for i in range(k+1, n):
            m = A[i, k] / A[k, k]
            A[i, k+1:] -= m * A[k, k+1:]
            b[i] -= m * b[k]
    
    x = zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x


def subti_forward(U : list[list[Union[int, float]]],
                  b : list[Union[int, float]]) -> list[Union[int, float]]:
    x = [0] * len(b)
    for i in range(len(b)-1, -1, -1):
        sum = b[i]
        for j in range(i+1, len(b)):
            sum -= U[i][j] * x[j]
        x[i] = sum / U[i][i]
    return x


def copy(A : Union[int, float]) -> list[list[Union[int, float]]]:
    """
    Fungsi untuk melakukan copy matriks


    """
    res = []
    for i in range(len(A)):
        row = []
        for j in range(len(A)):
            row.append(A[i][j])
        res.append(row)
    return res

def LU(A : list[list[Union[int, float]]]) -> list[list[Union[int, float]]]:
    """
    Fungsi Untuk membuat segitiga atas dan segitiga bawah

    parameter:
    A : Matriks

    return:
    L : segitiga atas
    U : segitiga bawah
    
    """
    n = len(A)
    L = identity(n)
    U = copy(A)

    for k in range(n-1):
        for i in range(k+1, n):
            if U[k, k] == 0:
                raise ValueError("Pivot nol ditemukan.Faktorisasi LU tidak dapat dilakukan.")
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k, n):
                U[i, j] -= L[i, k] * U[k, j]
    return L, U

def penjumlahan(
        A : list[list[Union[int, float]]],
        B : list[list[Union[int, float]]]
) -> list[list[Union[int, float]]]:
    """
    Fungsi untuk melakukan kalkulasi penjumlahan 2 matriks

    Parameter :
    A(int, float) : Matriks pertama
    B(int, float) : Matriks kedua

    return :
    result(int, float) : hasil kalkulasi
    """
    if len(A) != len(B):
        return error.IndeksError("Ukuran tidak sama")
    else:
        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(A)):
                op = A[i][j] + B[i][j]
                row.append(op)
            result.append(op)
        return result
    

def pengurangan(
        A : list[list[Union[int, float]]],
        B : list[list[Union[int, float]]]
) -> list[list[Union[int, float]]]:
    """
    Fungsi untuk melakukan kalkulasi pengurangan 2 matriks

    Parameter :
    A(int, float) : Matriks pertama
    B(int, float) : Matriks kedua

    return :
    result(int, float) : hasil kalkulasi
    """
    if len(A) != len(B):
        return ValueError("Ukuran Matriks Harus Sama")
    else:
        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(A)):
                op = A[i][j] - B[i][j]
                row.append(op)
            result.append(op)
        return result


def perkalian(
        A : list[list[Union[int, float]]],
        B : list[list[Union[int, float]]]
) -> list[list[Union[int, float]]]:
    """
    Fungsi untuk melakukan kalkulasi perkalian 2 matriks

    Parameter :
    A(int, float) : Matriks pertama
    B(int, float) : Matriks kedua

    return :
    result(int, float) : hasil kalkulasi
    """
    baris1 = len(A)
    kolom1 = len(A[0])
    baris2 = len(B)
    kolom2 = len(B[0])
    if kolom1 != baris2:
        return error.IndeksError("kolom dan baris harus sama")
    result = zeros(baris1, kolom2)
    for i in range(baris1):
        for j in range(kolom2):
            for k in range(kolom1):
                result[i][j] += A[i][k] * B[k][j]
    return result

def perkalian_skalar_matriks(
        matriks : list[list[Union[int, float]]],
        x : Union[int, float]
) -> list[list[Union[int, float]]]:
    """
    Fungsi untuk melakukan kalkulasi perkalian skalar matriks

    Parameter :
    Matriks(int, float) : Matriks
    x(int, float) : skalar

    return :
    result(int, float) : hasil kalkulasi
    """
    result = []
    n = len(matriks)
    for i in range(n):
        row = []
        for j in range(n):
            multi = x * matriks[i][j]
            row.append(multi)
        result.append(row)
    return result


if __name__ == "__main__":
    pass
