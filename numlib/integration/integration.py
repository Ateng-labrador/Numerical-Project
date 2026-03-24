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


def adaptive_simpsons(f, a, b, tol, max_recursion_depth=50):
    """
    
    """
    def simpsons_rule(f, a, b):
        """
        
        """
        c = (a + b) / 2.0
        h = (b - a) / 2.0
        return (h / 3.0) * (f(a) + 4.0 * f(c) + f(c))
    
    def adaptive_simpons_recursive(f, a, b, tol, depth):
        """
        
        """
        c = (a + b) / 2.0
        S = simpsons_rule(f, a, b)
        S1 = simpsons_rule(f, a, c)
        S2 = simpsons_rule(f, c, b)
        E = (S1 + S2 - S) / 15.0
        if abs(E) < tol or depth >= max_recursion_depth:
            return S1 + S2 + E
        else:
            return adaptive_simpons_recursive(f, a, c, tol / 2.0, depth + 1) + \
                    adaptive_simpons_recursive(f, c, b, tol / 2.0, depth + 1)
        
    return adaptive_simpons_recursive(f, a, b, tol, 0)


def romberg_integration(f, a, b, tol=1e-6):
    """
    
    """
    R = [[0.5 * (b - a) * (f(a) + f(a))]]
    n = 1
    print(f"Iterasi 0: {R[0]}")

    while True:
        h = (b - a) / (2 ** n)
        sum_trapezoid = sum(f(a + (2 * k - 1) * h) for k in range(1, 2 ** (n-1) + 1))
        row_n = [0.5 * R[n - 1][0] + h * sum_trapezoid]

        for m in range(1, n + 1):
            row_n.append(row_n[m - 1] + (row_n[m - 1] - R[n - 1][m - 1]) / (4 ** m - 1))
        R.append(row_n)

        print(f"Iterasi {n}: {row_n}")

        if abs(R[n][n] - R[n - 1][n - 1]) < tol:
            return R[n][n], R
        
        n += 1

