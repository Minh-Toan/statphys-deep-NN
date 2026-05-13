import numpy as np
from numba import njit
import csv

import sys
import os
folder = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', '..', '..'))
sys.path.insert(0, folder)
import mcmc


d = 150
gamma = 0.5
Delta = 1.25
k = int(gamma*d)
vlaw='ones'
prior='rad'

@njit(cache=True)
def sig(x): 
    return (x**2-1)/np.sqrt(2) + (x**3-3*x)/6

alphas = [2.6, 2.8, 3.0, 3.2] 
ncycles = [100, 50, 50, 50] 
n_posterior_samples = 4

def main():
    with open('info.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        for i in range(n_posterior_samples):
            for alpha, ncycle in zip(alphas, ncycles):
                W0, X, Y, v = mcmc.data_generate(d, gamma, alpha, Delta, vlaw, sig)
                Ws = mcmc.mcmc(W0, X, Y, Delta, v, sig, ncycle, info=True)[-100:]
                mmses = mcmc.test_error(Ws, W0, v, sig, ntest=10000)
                mmse = np.mean(mmses)
                qws, q2s = mcmc.overlaps(Ws, W0, v)
                qw, q2 = np.mean(qws), np.mean(q2s)
                writer.writerow([alpha, mmse, qw, q2])
                file.flush()
                
if __name__ == "__main__":
    main()
