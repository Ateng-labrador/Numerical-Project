from numlib.util import error
from typing import Union


def factorial(n) -> Union[int, error.ErrorValue]:
    """
    Fungsi untuk melakukan kalkulasi faktorial

    n! = n * (n - 1) * (n - 2) * ... * 1

    Parameters:
    n (int) : bilangan yang akan di kalkulasi

    Returns:
    f (int) : hasil dari kalkulasi faktorial
    """
    if n < 0:
        return error.ErrorValue("Nilai tidak boleh di bawah nol")
    if n == 0:
        return 1
    else:
        f = 1
        for i in range(1, n + 1):
            f *= i
        return f


def kombinasi(n : int, r : int) -> Union[int, error.ErrorValue]:
    """
    Fungsi untuk melakukan kalkulasi kombinasi

    n! / k! (n - k)!

    
    Parameters:
    n (int) = bilangan yang ingin di kalkulasi
    r (int) = bilangan yang ingin di kalkulasi

    return:
    C = hasil fungsi kombinasi
    """
    if n < 0 or r < 0:
        return error.ErrorValue("Nilai tidak boleh kurang dari nol")
    if r > n:
        return 0
    else:
        penyebut = factorial(n)
        pembilang = factorial(r) * factorial(n - r)
        C = penyebut / pembilang
        return C


def permutasi(n : int, r : int) -> int:
    """
    Fungsi untuk melakukan kalkulasi kombinasi

    n! / (n - k)!

    
    Parameters:
    n (int) = bilangan yang ingin di kalkulasi
    r (int) = bilangan yang ingin di kalkulasi

    return:
    C = hasil fungsi permutasi
    """
    if n <= 0 or r <= 0:
        return error.ErrorValue("Nilai tidak boleh kurang dari nol")
    else:
        penyebut = factorial(n)
        pembilang = factorial(n - r)
        return penyebut / pembilang
