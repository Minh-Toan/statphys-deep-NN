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
    return (x**2-1)/np.sqrt(2)

alphas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4] 
ncycles = [50, 50, 50, 50, 100, 100, 200] 
n_posterior_samples = 3

def main():
    with open('info.csv', 'a', newline='') as file1, open('unfo.csv', 'a', newline='') as file2:
        writer1 = csv.writer(file1)
        writer2 = csv.writer(file2)
        for i in range(n_posterior_samples):
            for alpha, ncycle in zip(alphas, ncycles):
                W0, X, Y, v = mcmc.data_generate(d, gamma, alpha, Delta, vlaw, sig)
                Ws_info = mcmc.mcmc(W0, X, Y, Delta, v, sig, ncycle, info=True)[-100:]
                Ws_unfo = mcmc.mcmc(W0, X, Y, Delta, v, sig, ncycle, info=False)[-100:]
                mmses_i = mcmc.test_error(Ws_info, W0, v, sig, ntest=10000)
                mmses_u = mcmc.test_error(Ws_unfo, W0, v, sig, ntest=10000)
                mmse_i, mmse_u = np.mean(mmses_i), np.mean(mmses_u)
                qws_i, q2s_i = mcmc.overlaps(Ws_info, W0, v)
                qws_u, q2s_u = mcmc.overlaps(Ws_unfo, W0, v)
                qw_i, q2_i = np.mean(qws_i), np.mean(q2s_i)
                qw_u, q2_u = np.mean(qws_u), np.mean(q2s_u)
                writer1.writerow([alpha, mmse_i, qw_i, q2_i])
                writer2.writerow([alpha, mmse_u, qw_u, q2_u])
                file1.flush()
                file2.flush()
if __name__ == "__main__":
    main()
