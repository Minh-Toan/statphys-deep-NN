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
    return (x**3-3*x)/np.sqrt(6)

alphas = [0.2, 0.4, 0.6, 0.8, 1.0] 
ncycles = [100, 100, 150, 150, 150] 
info = False
filename = 'unfo.csv'
n_posterior_samples = 9

def save_with_version(filename_base, data):
    version = 1
    filename = f"{filename_base}_{version}.csv"
    while os.path.exists(filename):
        version += 1
        filename = f"{filename_base}_{version}.csv"
    np.savetxt(filename, data, delimiter=',')

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
                save_with_version(f'single_runs/mmses_u_{alpha}', mmses_)
                
if __name__ == "__main__":
    main()
