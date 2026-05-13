import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, quad_vec
from scipy.optimize import fsolve

''' 
    Model
    Y = W.T D_v W/k + sqrt(t)*Z
    W Gaussian size kxd, k/d = a
    Z GOE with supp [-2,2]
'''

def stieltjes_ones(x, t, a): 
    p = [-t, -a*t - x, -a*x + a - 1, -a]
    sols = np.roots(p)
    choice = np.argmax(np.imag(sols))
    return sols[choice]

def supp_ones(t, a):
    p = [a**2,
    -2*a**3*t - 2*a**2 - 2*a,
    a**4*t**2 + 8*a**3*t - 2*a**2*t + a**2 - 2*a + 1,
    -2*a**4*t**2 + 8*a**3*t**2 - 10*a**3*t + 2*a**2*t + 8*a*t,
    -4*a**4*t**3 + a**4*t**2 - 20*a**3*t**2 + 4*a**3*t - 8*a**2*t**2 - 12*a**2*t + 12*a*t - 4*t]
    sols = np.roots(p)
    criteria = np.abs(np.imag(sols))<1e-6
    edges = np.real(sols[criteria])
    return np.min(edges), np.max(edges)

def stieltjes_rad(x, t, a):
    p = [t, x, -a**2*t - a + 1, -a**2*x, -a**2]
    sols = np.roots(p)
    choice = np.argmax(np.imag(sols))
    return sols[choice]

def supp_rad(t, a):
    p=[a**4,
    -2*a**6*t**2 + 5*a**5*t - 8*a**4*t + a**4/4 - 5*a**3 - 2*a**2,
    a**8*t**4 + 3*a**7*t**3 + 12*a**6*t**3 + 3*a**6*t**2 - 13*a**5*t**2 + \
    a**5*t + 22*a**4*t**2 - 26*a**4*t + 13*a**3*t - a**3 + 12*a**2*t + 3*a**2 - 3*a + 1,
    -4*a**8*t**5 - 16*a**7*t**4 - 16*a**6*t**4 - 24*a**6*t**3 - 16*a**5*t**3 - 16*a**5*t**2 \
    - 24*a**4*t**3 + 16*a**4*t**2 - 4*a**4*t + 16*a**3*t**2 + 16*a**3*t - 16*a**2*t**2 - 24*a**2*t + 16*a*t - 4*t]
    sols = np.roots(p)
    criteria = np.abs(np.imag(sols))<1e-6
    edges = np.real(sols[criteria])
    return  -np.sqrt(np.max(edges)), np.sqrt(np.max(edges))

def stieltjes_4point(x, t, a):
    p = [-9*t, -9*x, 50*a**2*t + 9*a - 9, 50*a**2*x, -25*a**4*t - 25*a**3 + 50*a**2, -25*a**4*x, -25*a**4]
    sols = np.roots(p)
    choice = np.argmax(np.imag(sols))
    return sols[choice]

def supp_4point(t, a): # works for a=0.5
    return  -5-2*np.sqrt(t), 5+2*np.sqrt(t)

def stieltjes_gauss(x, t, a):
    def R(s): 
        func = lambda x: a*x/(a-s*x)*np.exp(-x**2/2)
        return quad_vec(func, -10, 10)[0]/np.sqrt(2*np.pi) + t*s
    def eq(g): # equation satisfied by Stieltjes transform g(x)
        g = g[0] + 1j*g[1]
        eq = R(-g) - 1/g - x
        return [eq.real, eq.imag]
    eps = 1e-5
    init = [-eps, eps] if x>=0 else [eps, eps]
    sol = fsolve(eq, init)
    return sol[0] + abs(sol[1])*1j
    
def supp_gauss(t, a):
    return  -5-2*np.sqrt(t), 5+2*np.sqrt(t)

def stieltjes_unif(x, t, a):
    R = lambda s: -a/s + 1/(2*np.sqrt(3))*a**2/(s**2)*(np.log(a+np.sqrt(3)*s)-np.log(a-np.sqrt(3)*s)) + t*s # R-transform
    def eq(g): # equation satisfied by Stieltjes transform g(x)
        g = g[0] + 1j*g[1]
        eq = R(-g) - 1/g - x
        return [eq.real, eq.imag]
    eps = 1e-5
    init = [-eps, eps] if x>=0 else [eps, eps]
    sol = fsolve(eq, init)
    return sol[0] + abs(sol[1])*1j
    
def supp_unif(t, a):
    return  -4.5-2*np.sqrt(t), 4.5+2*np.sqrt(t)   

stieltjes_map = {
'ones': stieltjes_ones,
'rad': stieltjes_rad, 'radd': stieltjes_rad,
'unif': stieltjes_unif, 'unid': stieltjes_unif,
'gauss': stieltjes_gauss, 'gaussd': stieltjes_gauss,
'4point': stieltjes_4point
}

supp_map = {
'ones': supp_ones,
'rad': supp_rad, 'radd': supp_rad,
'unif': supp_unif, 'unid': supp_unif,
'gauss': supp_gauss, 'gaussd': supp_gauss,
'4point': supp_4point
}

def stieltjes(x, t, a, vlaw):   
    return stieltjes_map[vlaw](x, t, a)

def supp(t, a, vlaw):
    return supp_map[vlaw](t, a)

def f_RIE(R, t, a, vlaw):
    if t<1e-6:
        return R
    h = lambda x: -stieltjes(x+1e-6j, t, a, vlaw).real
    eigval, eigvec = np.linalg.eigh(R)
    eigval_denoised = np.array([e - 2*t*h(e) for e in eigval])
    return eigvec@np.diag(eigval_denoised)@eigvec.T

def F_RIE(t, a, vlaw):
    xmin, xmax = supp(t, a, vlaw)
    rho = lambda x: stieltjes(x, t, a, vlaw).imag/np.pi
    return t - 4*np.pi**2/3 * t**2 * quad(lambda x: rho(x)**3, xmin, xmax, epsabs=1e-13, epsrel=1e-13)[0]
