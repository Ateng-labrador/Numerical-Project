from colorama import Fore, Style

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