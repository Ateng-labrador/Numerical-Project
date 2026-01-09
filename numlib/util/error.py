from colorama import Fore, Style

#
# mendeteksi erorr terkait indeks
# membuat error kostum
# membuat value error
# menampilkan error karna di bagi dengan nol
#


# default error dari warna menggunakan kode ANSI
red: str = Fore.RED
reset_warna: str = Style.RESET_ALL

class IndeksError(IndexError):
    """
    Kelas untuk membuat error dari index jika tidak
    selaras antar dimensi.

    Parameter:
        pesan (str): pesan kostum
    """
    def __init__(self, pesan : str):
        message = f"{red}Error:{reset_warna} {pesan}"
        super().__init__(message)


class Error(Exception):
    """
    Kelas untuk kostomisasi error 

    parameter:
        pesan (str): pesan kostum
    """
    def __init__(self, pesan: str):
        message = f"{red}Error:{reset_warna} {pesan}"
        super().__init__(message)

class ErrorValue(ValueError):
    """
    Kelas untuk membuat error dari index dengan throw ValueError

    parameter:
        pesan (str): pesan kostum
    """
    def __init__(self, pesan: str):
        message = f"{red}Error: {reset_warna} {pesan}"
        super().__init__(message)


class ErrorDibagiNol(ZeroDivisionError):
    """
    Kelas untuk menampilkan error yang tidak bisa dibagi dengan nol
    """
    def __init__(self):
        super().__init__(
            f"{red}Error Dibagi Nol:{reset_warna} Tidak bisa dibagi dengan nol"
        )


class ErrorTipeData(TypeError):
    pass
