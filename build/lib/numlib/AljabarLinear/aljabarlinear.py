import numpy as np
import math
from numlib.util import error
from numlib.special import *
from typing import Union


def _array_square(A):
    """
    Mengubah menjadi array
    """
    A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('input nya tidak sesuai array')
    return A

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


def zeros(n : int , m : int = None) -> list[list[int]]:
    """
    Fungsi untuk membuat matriks nol

    parameter:
    n : kolom
    m : baris (None)

    return:
    matriks : matriks nol
    """
    matriks = []
    if m == None:
        for _ in range(n):
            matriks.append(0)
        return matriks
    else:
        for _ in range(n):
            row = []
            for _ in range(m):
                row.append(0)
            matriks.append(row)
        return matriks


def dft(n, scale=None):
    """
    e^(-2pi*i*k*n/ N)
    """
    omegas = np.exp(-2j * np.pi * np.arange(n) / n).reshape(-1, 1)
    m = omegas ** np.arange(n)
    if scale == 'sqrt':
        m /= math.sqrt(n)
    elif scale == 'n':
        m /= n
    return m


def pascal(n, kind='symmetric', exact=True):
    """
    
    """
    if kind not in ['symmetric', 'lower', 'upper']:
        return error.Error("Tipe harus 'symmetric', 'lower', atau 'upper'")
    L_n = []
    for i in range(n):
        colum = []
        for j in range(n):
            x = kombinasi(i, j)
            colum.append(x)
        L_n.append(colum)
    x = _array_square(L_n)
    if kind.lower() == 'lower':
        return x
    elif kind.lower() == 'upper':
        return x.T
    else:
        return np.dot(x , x.T)


def pauli_matriks(A):
    """
    matriks pauli
    """

    if A == "x":
        colum = []
        for i in range(2):
            row = []
            for j in range(2):
                if i == j:
                    row.append(0)
                else:
                    row.append(1)
            colum.append(row)
        return np.array(colum)

    elif A == "z":
        colum = []
        for i in range(2):
            row = []
            for j in range(2):
                if i == j:
                    row.append(1)
                else:
                    row.append(0)
            colum.append(row)
        return np.array(colum)

    elif A == "y":
        colum = []
        for i in range(2):
            row = []
            for j in range(2):
                if i == j:
                    row.append(1)
                else:
                    row.append(0)
            colum.append(row)
        return np.array(colum)


def hankel_matriks(A, b=1):
    res = []
    for i in range(b, A+1):
        row = []
        for j in range(b, A+1):
            row.append(i + j - 1)
        res.append(row)
    return np.array(res)


def hilbert_matriks(A, b=1):
    res = []
    for i in range(b, A+1):
        row = []
        for j in range(b, A+1):
            row.append(1/(i + j - 1))
        res.append(row)
    return np.array(res)


def toeplite_matriks(A, b=1):
    res = []
    for i in range(b, A+1):
        row = []
        for j in range(b, A+1):
            row.append((i - 1))
        res.append(row)
    return np.array(res)



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


def transpose(A: list[list[Union[int, float]]]) -> list[list[Union[int, float]]]:
    """
    
    """
    x = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
    return x


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


#*
# matriks khusus
# matriks simetris
# matriks definit positif
# matriks pita (banded matrix)
# *#

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


def LDLT_decomposition(A : list[list[Union[int, float]]]) -> list[list]:
    """
    Fungsi Untuk Melakukan faktorisasi LDLT composition
    A = LDL^T

    Parameter :
    A = matriks non-singular

    return :
    L = matriks segitiga bawah
    d = matriks diagonal
    Lt = matriks segitiga bawah yang di transposekan
    """
    if transpose(A) == A:
        n = len(A)
        L = zeros(n, n)
        d = zeros(n)
        for i in range(n):
            sum_val = 0.0
            for k in range(i):
                sum_val += (L[i][k] ** 2) * d[k]
            d[i] = A[i][i] - sum_val

            for j in range(i + 1, n):
                sum_val = 0.0
                for k in range(i):
                    sum_val += L[j][k] * L[i][k] * d[k]
                L[j][i] = (A[j][i] - sum_val) / d[i]
            L[i][i] = 1

        Lt = [[L[j][i] for j in range(n)] for i in range(len(A[0]))]
        return L, d, Lt
    else:
        return error.Error("Matriks Tidak simetri")


def cholesky_decomposition(A : list[list[Union[int, float]]]) -> list[list[Union[int, float]]]:
    """
    Fungsi untuk melakukan decomposition cholesky
    yang akan membuat faktorisasi A = G * Gt

    Parameter:
    A = matriks non-singular dan simetri
    return:
    G = matriks segitiga bawah
    Gt = matriks segitiga bawah yang di trasnpose
    """
    if transpose(A) == A:
        n = len(A)
        G = zeros(n, n)
        for i in range(n):
            for j in range(i + 1):
                sum = 0
                if j == i:
                    for k in range(j):
                        sum += G[j][k] * G[j][k]
                    G[j][j] = (A[j][j] - sum)**(1/2)
                else:
                    for k in range(j):
                        sum += G[i][k] * G[j][k]
                    G[i][j] = (A[i][j] - sum) // G[j][j]
        Gt = transpose(G)
        return G, Gt
    else :
        return error.Error("Matriksnya tidak simetri")


def LU_tridiagonal(n, A):
    """
    Masih perlu di tinjau
    """
    L = np.zeros((n, n))
    U = np.eye(n)
    l11 = A[0, 0]
    u12 = A[0, 1] / l11
    z1 = A[0, -1] / l11

    L[0, 0] = l11
    U[0, 1] = u12
    z = np.zeros(n)
    z[0] = z1
    for i in range(1, n-1):
        L[i, i-1] = A[i, i-1]
        L[i, i] = A[i, i] - L[i, i-1] * U[i-1, i]
        U[i, i+1] = A[i, i+1] / L[i, i]
        z[i] = (A[i, -1] - L[i, i-1] * z[i-1]) / L[i, i]
    
    L[n-1, n-2] = A[n-1, n-2]
    L[n-1, n-1] = A[n-1, n-1] - L[n-1, n-2] * U[n-2, n-1]
    z[n-1] = (A[n-1, -1] - L[n-1, n-2] * z[n-2]) / L[n-1, n-1]

    x = np.zeros(n)
    x[n-1] = z[n-1]

    for i in range(n-2, -1, -1):
        x[i] = z[i] - U[i, i+1] * x[i+1]
    return x, L, U


def rotation(matriks : list[list[Union[int, float]]],
             rotasi = 90) -> list[list[Union[int, float]]]:
    """
    Fungsi untuk melakukan rotasi 90 dan 180.

    parameter:
    A = matriks
    rotasi = 90 dan 180

    """
    if rotasi == 90:
        baris1 = len(matriks[0])
        kolom1 = len(matriks)
        # reverse the order of rows
        matriks = matriks[::-1]
        result = zeros(kolom1, baris1)

        for i in range(baris1):
            for j in range(kolom1):
                result[i][j] = matriks[j][i]
        return result
    if rotasi == 180:
        # reverse the order of rows
        # matriks pertama
        matriks = matriks[::-1]
        result1 = zeros(kolom1, baris1)
        # matriks kedua
        matriks1 = result1[::-1]
        result_f = zeros(kolom1, baris1)

        for i in range(baris1):
            for j in range(kolom1):
                result[i][j] = matriks[j][i]
        for i in range(baris1):
            for j in range(kolom1):
                result_f[i][j] = matriks1[j][i]
        return result_f
    else:
        return error.IndeksError("Indeks tidak terdefinisi")
