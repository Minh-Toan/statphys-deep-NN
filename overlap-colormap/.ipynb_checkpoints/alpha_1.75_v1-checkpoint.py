import numpy as np
import tensorflow as tf
import traceback, os, gc

tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[1], 'GPU') ###

import hmc_2hl as hmc

d = 200
alpha = 1.75
gamma_1 = 0.5
gamma_2 = 0.5
Delta = 0.2
vlaw = 'gaussd'

sig_1 = lambda x: tf.nn.tanh(2*x) / 0.797032
sig_2 = lambda x: tf.nn.tanh(2*x) / 0.797032

params = {
    'step_size': 0.01,
    'num_leapfrog_steps': 10,
    'num_adaptation_steps': 50000, ###
}

out_dir = f'alpha_{alpha}_longer_HMC'
os.makedirs(out_dir, exist_ok=True)

for i in range(2, 100): ###
    try:
        tf.keras.backend.clear_session()
        gc.collect()

        W10, W20, v, X, Y = hmc.data_generate(d, alpha, gamma_1, gamma_2, sig_1, sig_2, Delta, vlaw)
        W1s_info, W2s_info = hmc.hmc(params, W10, W20, v, X, Y, Delta, sig_1, sig_2, info=True)

        W1a, W2a = W1s_info[-1], W2s_info[-1]
        np.savetxt(f'{out_dir}/W1a_{i+1}.csv', W1a, delimiter=',')
        np.savetxt(f'{out_dir}/W10_{i+1}.csv', W10.numpy(), delimiter=',')
        np.savetxt(f'{out_dir}/W2a_{i+1}.csv', W2a, delimiter=',')
        np.savetxt(f'{out_dir}/W20_{i+1}.csv', W20.numpy(), delimiter=',')

        del W1s_info, W2s_info, W10, W20, W1a, W2a, X, Y, v
        gc.collect()
    except Exception:
        with open("err.txt", "a") as f:
            f.write(f"Error in loop {i}:\n{traceback.format_exc()}\n\n")
        tf.keras.backend.clear_session()
        gc.collect()
        continue
