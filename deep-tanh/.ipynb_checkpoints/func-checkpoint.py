import numpy as np
import math
from scipy.integrate import quad


'''
gaussian integrals
'''
def Ez(f):
    return quad(lambda x: f(x)*np.exp(-x**2/2)/np.sqrt(2*np.pi), -10, 10)[0]

def h(q, sig): # E[sig(u) sig(v)], where u,v~N(0,1), E[uv]=q
    f = lambda u: Ez( lambda v: sig(np.sqrt(q)*u + np.sqrt(1-q)*v) )
    f = np.vectorize(f)
    return Ez( lambda u: f(u)**2 )

'''
activation functions and Hermite coefficients
'''
def hermite(x, n):
    if n==0:
        return 1
        
    a, b = 1, x
    for k in range(2, n + 1):
        a, b = b, x*b - (k - 1)*a
    return b


def hermite_coeffs(f, n):
    return np.array([Ez(lambda x: f(x)*hermite(x, i)) for i in range(n+1)])

def gg(sig, num_coeff):
    coeffs = hermite_coeffs(lambda x: sig(x), num_coeff)
    # special handling for relu
    x = np.array([-2, -1, 0, 1, 2])
    if np.allclose([sig(xi) for xi in x], np.maximum(x, 0)):
        def g(x):
            if x==1:
                return 1/4*(1 - 3/(np.pi))
            return -1/(2*np.pi) - x**2/(4*np.pi) + np.sqrt(1-x**2)/(2*np.pi) + x*np.arctan(x/np.sqrt(1-x**2))/(2*np.pi)
        def g_prime(x):
            if x==1:
                return 1/4 - 1/(2*np.pi)
            return (np.arctan(x/np.sqrt(1-x**2))-x)/(2*np.pi)
        return g, g_prime, coeffs
    # other activations
    g = lambda x: sum((coeffs[i]**2*x**i/math.factorial(i)) for i in range(3, len(coeffs)))
    g_prime = lambda x: sum(coeffs[i]**2*x**(i-1)/math.factorial(i-1) for i in range(3, len(coeffs)))
    return g, g_prime, coeffs