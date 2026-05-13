import numpy as np
import csv
import algo
import rie_denoising as rie



d = 200
gamma = 0.5
Delta = 0.1
k = int(gamma*d)
sig1 = lambda x: np.maximum(x,0)
sig2 = lambda x: np.tanh(2*x)
prior = 'gauss'
alphas = np.arange(0.125, 8.125 + 1e-9, 0.25)

# sigs = [sig1, sig2]
# sig_names = ['relu', 'tanh']
sigs = [sig2]
sig_names = ['tanh']
vlaws = ['ones', '4point', 'gauss']


for vlaw in vlaws:
    f_RIE = lambda R, t: rie.f_RIE(R, t, gamma, vlaw)
    F_RIE = lambda t: rie.F_RIE(t, gamma, vlaw)
    for sig, sig_name in zip(sigs, sig_names):
        with open(f'data_fig_5/{sig_name}_{vlaw}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            for i in range(12):
                mmses = [algo.algo_perf(d, gamma, alpha, Delta, f_RIE, F_RIE, sig, vlaw, prior) for alpha in alphas]
                writer.writerow(mmses)
                file.flush()

