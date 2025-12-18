import numpy as np
import tensorflow as tf
import traceback, os, gc

tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[1], 'GPU') ###

import hmc_2hl as hmc

d = 200
alpha = 2.75
gamma_1 = 0.5
gamma_2 = 0.5
Delta = 0.2
vlaw = 'gaussd'

sig_1 = lambda x: tf.nn.tanh(2*x) / 0.797032
sig_2 = lambda x: tf.nn.tanh(2*x) / 0.797032

params = {
    'step_size': 0.01,
    'num_leapfrog_steps': 10,
    'num_adaptation_steps': 25000, ###
}

n_episode = 6 ###

out_dir = f'alpha_{alpha}_v2' ###
os.makedirs(out_dir, exist_ok=True)

for i in range(100): ###
    tf.keras.backend.clear_session()
    gc.collect()
    
    W10, W20, v, X, Y = hmc.data_generate(d, alpha, gamma_1, gamma_2, sig_1, sig_2, Delta, vlaw)
    W1_init = W10
    W2_init = W20
    
    for j in range(n_episode):
        W1s, W2s = hmc.hmc(params, W1_init, W2_init, v, X, Y, Delta, sig_1, sig_2)
        W1_init = W1s[-1]
        W2_init = W2s[-1]

    W1a, W2a = W1s[-1], W2s[-1]
    np.savetxt(f'{out_dir}/v_{i+1}.csv', v, delimiter=',')
    np.savetxt(f'{out_dir}/W1a_{i+1}.csv', W1a, delimiter=',')
    np.savetxt(f'{out_dir}/W10_{i+1}.csv', W10.numpy(), delimiter=',')
    np.savetxt(f'{out_dir}/W2a_{i+1}.csv', W2a, delimiter=',')
    np.savetxt(f'{out_dir}/W20_{i+1}.csv', W20.numpy(), delimiter=',')

    del W1s, W2s, W10, W20, W1a, W2a, X, Y, v
    gc.collect()
    
