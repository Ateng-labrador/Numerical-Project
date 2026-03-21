import numpy as np

# integral numerik

def rectangula_method(f, a, b, n, method = "midpoint"):
    """
    
    """
    if  method == "midpoint":
        h = (b - a) / n
        integral = sum(f(a + (i + 0.5)  * h) for i in range(n)) 
        return h * integral    
    elif method == "right":
        h = (b - a) / n
        integral = sum(f(a + (i + 1)  * h) for i in range(n)) 
        return h * integral
    elif method == "left":
        h = (b - a) / n
        integral = sum(f(a + i  * h) for i in range(n)) 
        return h * integral    
    else:
        KeyError("Error")


def trapezoid_rule(f, a, b, n):
    """
    
    """
    h = (b - a) / n
    integral = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        integral += f(a + i * h)
    integral *= h
    return  integral


def simpson(f, a, b, n, method = "38"):
    """
    
    """
    if method == "38":
        h = (b - a) / n
        integral = f(a) + f(b)

        for i in range(1, n):
            if i % 3 == 0:
                integral += 2 * f(a + i * h)
            else:
                integral += 3 * f(a + i * h)
        integral *= 3 * h/8
        return integral
    elif  method == "13":
        h = (b - a)/ n
        integral = f(a) + f(b)

        for i in range(1, n, 2):
            integral += 4 * f(a + i * h)
        for i in range(2, n, 2):
            integral += 2 * f(a + i * h)
        integral *= h/3
        return integral
    else:
        KeyError("Error")


def gauss_legendre_quadrature(f, a, b, n):
    """
    
    """
    if n == 2:
        x = [-1/np.sqrt(3), 1/np.sqrt(3)]
        w = [1, 1]
    elif n == 3:
        x = [-np.sqrt(3/5), 0, np.sqrt(3/5)]
        w = [5/9, 8/9, 5/9]
    else:
        raise ValueError("Jumlah Titik harus 2 atau 3")
    
    integral = 0.0
    for i in range(n):
        xi = 0.5 * ((b-a) * x[i] + (b + a))
        integral += w[i]* f(xi)
    
    integral *= 0.5 * (b - a)
    return integral
