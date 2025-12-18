import numpy as np
import tensorflow as tf
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU') ###

import csv
import hmc_1hl as hmc  ###

L = 1 ###
alphas = np.array([1+1/6, 1+2/6, 1+3/6, 1+4/6, 1+5/6, 2, 2+1/3, 2+2/3, 3, 3+1/3, 3+2/3, 4]) * L
nsteps = np.array([4000, 6000] + [2500]*10)

d = 50
gamma = 1
Delta = 0.1
vlaw = 'ones'
k = int(gamma * d)

sig = lambda x: (tf.nn.tanh(2*x) - 0.729477*x) / 0.321129
ntest = 10000
Xtest = tf.random.normal((d, ntest), dtype=tf.float32)

with open(f'deep_tanh_hmc_L_{L}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(9):
        mmses = []
        for alpha, nstep in zip(alphas, nsteps):
            params = {
                'step_size': 0.01,
                'num_leapfrog_steps': 10,
                'num_adaptation_steps': nstep,
            }
            W0, v, X, Y = hmc.data_generate(d, alpha, gamma, sig, Delta, vlaw)
            Ws_info = hmc.hmc(params, W0, v, X, Y, Delta, sig, info=True)
            test_info = hmc.test_error(Ws_info, W0, v, sig, Xtest)
            mmses.append(np.mean(test_info[-500:]))
        writer.writerow(mmses)
        file.flush()