import numpy as np


def forward_difference(f, x, h=1e-5):
    """
    
    """
    return (f(x + h) - f(x)) / h


def backward_difference(f, x, h=1e-5):
    """
    
    """
    return (f(x) - f(x - h)) / h


def central_difference(f, x, h=1e-5):
    """
    
    """
    return (f(x + h) - f(x - h)) / h


def second_derivative(f, x, h=1e-5):
    """
    
    """
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2


def euler_method(f, y0, t0, T, h):
    N = int((T - t0) / h)
    t = np.linspace(t0, T, N+1)
    y = np.zeros(N + 1)
    y[0] = y0

    for n in range(N):
        y[n+1] = y[n] + h * f(t[n], y[n])

    return t, y
