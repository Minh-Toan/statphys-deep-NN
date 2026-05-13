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

alphas = [1.3] 
ncycles = [2000] 
info = False
filename = 'unfo.csv'
n_posterior_samples = 3

def main():
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for i in range(n_posterior_samples):
            for alpha, ncycle in zip(alphas, ncycles):
                W0, X, Y, v = mcmc.data_generate(d, gamma, alpha, Delta, vlaw, sig)
                Ws_ = mcmc.mcmc(W0, X, Y, Delta, v, sig, ncycle, info)
                Ws = Ws_[-100:]
                mmses = mcmc.test_error(Ws, W0, v, sig, ntest=10000)
                mmse = np.mean(mmses)
                qws, q2s = mcmc.overlaps(Ws, W0, v)
                qw, q2 = np.mean(qws), np.mean(q2s)
                writer.writerow([alpha, mmse, qw, q2])
                file.flush()
                
                mmses_ = mcmc.test_error(Ws_, W0, v, sig, ntest=10000)
                np.savetxt(f'mmses_{alpha}_u_{i}.csv', mmses_, delimiter=',')
                
if __name__ == "__main__":
    main()
