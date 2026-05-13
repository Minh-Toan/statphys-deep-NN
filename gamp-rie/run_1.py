import numpy as np
import csv
import algo
import rie_denoising as rie



d = 150
gamma = 0.5
Delta = 0.1
k = int(gamma*d)
# sig = lambda x: np.maximum(x,0) # ReLU
sig = lambda x: np.tanh(2*x)
# sig = lambda x: np.where(x > 0, x, (np.exp(x) - 1)) # ELU 

vlaw = 'gauss'
prior = 'gauss'


alphas = [0.375, 1.375, 2.375, 3.375, 4.375, 5.375, 6.375]

f_RIE = lambda R, t: rie.f_RIE(R, t, gamma, vlaw)
F_RIE = lambda t: rie.F_RIE(t, gamma, vlaw)

with open(f'data_fig_5/tanh_{vlaw}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(9):
        mmses = [algo.algo_perf(d, gamma, alpha, Delta, f_RIE, F_RIE, sig, vlaw, prior) for alpha in alphas]
        writer.writerow(mmses)
        file.flush()

