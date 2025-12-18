import numpy as np
import tensorflow as tf
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')

import hmc_2hl as hmc
import csv

d = 200
alpha = 1.75
gamma_1 = 0.5
gamma_2 = 0.5

Delta = 0.2
vlaw = 'gaussd'
k1 = int(gamma_1 * d)
k2 = int(gamma_2 * d)

sig_1 = lambda x: tf.nn.tanh(2*x) / 0.797032
sig_2 = lambda x: tf.nn.tanh(2*x) / 0.797032

params = {
    'step_size': 0.01,
    'num_leapfrog_steps': 10,
    'num_adaptation_steps': 25000, ###
}


W10, W20, v, X, Y = hmc.data_generate(d, alpha, gamma_1, gamma_2, sig_1, sig_2, Delta, vlaw)

W1_init = W10
W2_init = W20

n_episode = 10 ###
nstep = params['num_adaptation_steps']*n_episode

f1 = f'small_test_data/qw1_alpha_{alpha}_step_{nstep}.csv'
f2 = f'small_test_data/qw2_alpha_{alpha}_step_{nstep}.csv'
with open(f1, 'w', newline='') as file1, open(f2, 'w', newline='') as file2:
    writer1 = csv.writer(file1)
    writer2 = csv.writer(file2)
    for i in range(n_episode):
        W1s, W2s = hmc.hmc(params, W1_init, W2_init, v, X, Y, Delta, sig_1, sig_2)
        qw1s = np.array([tf.reduce_mean(W1*W10) for W1 in W1s])
        qw2s = np.array([tf.reduce_mean(W2*W20) for W2 in W2s])
        writer1.writerow(qw1s)
        file1.flush()
        writer2.writerow(qw2s)
        file2.flush()
        W1_init = W1s[-1]
        W2_init = W2s[-1]
    



