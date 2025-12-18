import numpy as np
import math
from scipy.integrate import quad


'''
gaussian integral
'''
def Ez(f):
    return quad(lambda x: f(x)*np.exp(-x**2/2)/np.sqrt(2*np.pi), -10, 10)[0]

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

def gg(sig):
    num_coeff=15
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

'''
square root of matrices
'''
def sqrt_mat(A):
    D, U = np.linalg.eigh(A)
    D = np.sqrt(np.clip(D, 0, None)) # remove small negative numbers
    return U @ np.diag(D) @ U.T

'''
overlap function for channel y = sqrt(l)x+z, x~N(0, C), C has eigenvalues eigs
'''
def m_C(eigs, ls): # ls an array
    ls = np.array([ls]).T # convert ls into a nx1 array
    return np.sum(ls*eigs**2/(1+ls*eigs), axis=1)/len(eigs)


'''
handling real images
'''
def downsize_mnist(imgs):
    partitions = [4] + [2]*10 + [4]
    edges = np.cumsum([0] + partitions)
    # compute indices for all blocks
    rows = np.array([[i, j] for i in range(12) for j in range(12)])
    downsized = np.zeros((imgs.shape[0], 12, 12))
    for i in range(12):
        for j in range(12):
            downsized[:, i, j] = imgs[:, edges[i]:edges[i+1], edges[j]:edges[j+1]].mean(axis=(1,2))
    return downsized

def normalize(X): # normalize dataset X so that mean=0 and the covariance matrix C satisfies tr(C) = d
    d = len(X[0])
    X = X - X.mean(axis=0)
    C = np.cov(X, rowvar=False)
    scale = d/np.linalg.trace(C)
    C *= scale
    X *= np.sqrt(scale)
    return X, C

'''
histograms
'''
def empirical_density(data):
    counts, edges = np.histogram(data, bins=20, density=True)
    centers = (edges[:-1] + edges[1:])/2
    x_points = np.concatenate(([edges[0]], centers, [edges[-1]]))
    y_points = np.concatenate(([0], counts, [0]))
    return interp1d(x_points, y_points, kind='linear', bounds_error=False, fill_value=0.0)








    

