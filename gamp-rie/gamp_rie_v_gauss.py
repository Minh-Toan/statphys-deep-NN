import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad, quad_vec
from scipy.sparse.linalg import eigsh # computing smallest and largest eigenvalues
from scipy.interpolate import AAA
import csv

import algo


a = 0.5

d0 = 1000
k0 = int(a*d0)
v0 = np.mean(np.sort(np.random.randn(10000, k0)), axis=0)
W0 = np.random.randn(k0,d0)
Z0 = np.random.randn(d0,d0)
Z0 = (Z0+Z0.T)/np.sqrt(2*d0)
S0 = W0.T@np.diag(v0)@W0/k0

def supp(t):
    Y0 = S0 + np.sqrt(t)*Z0
    eigmin = eigsh(Y0, k=1, which='SA', return_eigenvectors=False)[0]
    eigmax = eigsh(Y0, k=1, which='LA', return_eigenvectors=False)[0]
    return eigmin, eigmax

def R(s): 
    func = lambda x: a*x/(a-s*x)*np.exp(-x**2/2)
    return quad_vec(func, -10, 10)[0]/np.sqrt(2*np.pi)

samples = np.linspace(-3, 3, 21) - 1e-6*1j
R_approx = AAA(samples, R(samples))

def stieltjes(x, t):
    def eq(g):
        g = g[0] + 1j*g[1]
        eq = R_approx(-g) - t*g - 1/g - x
        return [eq.real, eq.imag]
    eps = 1e-4
    init = [-eps, eps] if x>=0 else [eps, eps]
    sol = fsolve(eq, init)
    return sol[0] + abs(sol[1])*1j

def f_RIE(R, t):
    if t<1e-6:
        return R
    h = lambda x: -stieltjes(x-1e-6j, t).real
    eigval, eigvec = np.linalg.eigh(R)
    eigval_denoised = np.array([e - 2*t*h(e) for e in eigval])
    return eigvec@np.diag(eigval_denoised)@eigvec.T

def F_RIE(t):
    xmin, xmax = supp(t)
    rho = lambda x: stieltjes(x, t).imag/np.pi
    return t - 4*np.pi**2/3 * t**2 * quad(lambda x: rho(x)**3, xmin, xmax)[0]

d = 200
gamma = a 
Delta = 0.1
k = int(gamma*d)

sig = lambda x: np.maximum(x,0)

vlaw = 'gauss'
prior = 'gauss'

alphas = np.arange(0.125, 8.125 + 1e-9, 0.25)

with open(f'data_fig_5/relu_{vlaw}.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    for i in range(3):
        mmses = [algo.algo_perf(d, gamma, alpha, Delta, f_RIE, F_RIE, sig, vlaw, prior) for alpha in alphas]
        writer.writerow(mmses)
        file.flush()

