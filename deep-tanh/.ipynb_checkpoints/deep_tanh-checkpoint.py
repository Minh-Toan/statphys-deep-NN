import numpy as np
from scipy.interpolate import AAA
import func


qs = np.linspace(0, 1, 21)
gs = np.loadtxt('g_21.csv', delimiter=',')
g = AAA(qs, gs)

def K(q):
    L = len(q)
    Q = q[0]
    for l in range(1, L):
        Q = q[l]*g(Q)
    return g(Q)


def gradient(f, x, e1=1e-6, e2=1e-12):
    L = len(x)
    grad = np.zeros(L)
    for i in range(L):
        h = np.zeros(L)
        h[i] = 1
        if x[i]<e1: 
            grad[i] = (f(x+e2*h) - f(x))/e2
        elif x[i]>1-e1:
            grad[i] = (f(x) - f(x-e2*h))/e2
        else:
            grad[i] = (f(x+e1*h) - f(x-e1*h))/(2*e1) 
    return grad

def potential(alpha, Delta, q):
    return -alpha*np.log(Delta + 1 - K(q)) + np.sum( q + np.log(1-q) )

def quantities(q_init, alpha, Delta): 
    q = q_init.copy()
    for i in range(500):
        q_old = q.copy()
        mmse = 1 - K(q)
        r = alpha*gradient(K, q)/(Delta + mmse) 
        q = r/(r+1) 
        if np.max(np.abs(q - q_old))< 1e-12:
            break
    return np.insert(q, 0, mmse) 
    