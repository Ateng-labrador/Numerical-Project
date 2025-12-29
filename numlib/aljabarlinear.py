import numpy as np
import math 


def identity(n : int = 2):
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


def zeros(n : int = 2):
    """
    Fungsi untuk membuat matriks nol

    parameter:
    n : ukuran matris n x n

    return:
    matriks : matriks nol
    """
    matriks = []
    for _ in range(n):
        row = []
        for _ in range(n):
            row.append(0)
        matriks.append(row)
    return matriks


def dot(A, B):
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


def cross(A, B):
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
        return IndexError("Panjang melebihi")
    

def gauss_back_elim(A, b):
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


def subti_forward(U, b):
    x = [0] * len(b)
    for i in range(len(b)-1, -1, -1):
        sum = b[i]
        for j in range(i+1, len(b)):
            sum -= U[i][j] * x[j]
        x[i] = sum / U[i][i]
    return x


def LU(A):
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
    U = np.copy(A)

    for k in range(n-1):
        for i in range(k+1, n):
            if U[k, k] == 0:
                raise ValueError("Pivot nol ditemukan.Faktorisasi LU tidak dapat dilakukan.")
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k, n):
                U[i, j] -= L[i, k] * U[k, j]
    return L, U
