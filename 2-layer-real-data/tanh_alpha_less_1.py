import numpy as np
import tensorflow as tf
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[1], 'GPU')
from tensorflow.keras.datasets import mnist

import csv
import hmc
import func


def preprocess(X):
    X = func.downsize_mnist(X)
    X = X.reshape(len(X), -1)
    X, _ = func.normalize(X)
    return X

# load, downsize and normalize the dataset
(Xtrain, _), (Xtest, _) = mnist.load_data()
Xtrain = preprocess(Xtrain)
Xtrain = Xtrain[np.random.RandomState(0).permutation(len(Xtrain))].T
Xtest = preprocess(Xtest).T

Xtrain = tf.convert_to_tensor(Xtrain, dtype=tf.float32)
Xtest = tf.convert_to_tensor(Xtest, dtype=tf.float32)

d = 144
gamma = 0.5
Delta = 0.1
k = int(gamma*d)

vlaw = 'gaussd'
sig = lambda x: tf.nn.tanh(2*x)

alphas = [0.04, 0.125, 0.25, 0.375, 0.5, 2/3, 5/6] 
# alphas = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75] 

params = {'step_size': 0.01,
          'num_leapfrog_steps': 10,
          'num_adaptation_steps': 4000, ###
          'num_post_adapt_steps': 0}

with open(f'tanh_alpha_less_1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(9):
    # for i in range(9):
        mmses = []
        for alpha in alphas:
            n = int(alpha*d**2)
            X = Xtrain[:,:n]
            W0, Y, v = hmc.data_generate(d, k, n, Delta, sig, vlaw, X)
            Ws_info = hmc.hmc(params, W0, X, Y, v, gamma, alpha, Delta, sig, show_acceptance_rate=False, show_adaptation_steps=True)
            test_info =  hmc.test_error(Ws_info, W0, v, sig, Xtest)
            mmses.append(np.mean(test_info[-500:]))
        writer.writerow(mmses)
        file.flush()







