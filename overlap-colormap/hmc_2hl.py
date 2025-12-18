import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def pre_activation(v, sig_2, W2, sig_1, W1, X):
    k1, d = W1.shape
    k2, _ = W2.shape
    c_d, c_k1, c_k2 = 1/np.sqrt(d), 1/np.sqrt(k1), 1/np.sqrt(k2)
    M = W1 @ X * c_d
    M = sig_1(M)
    M = W2 @ M * c_k1
    M = sig_2(M)
    return tf.tensordot(v, M, axes=1) * c_k2

def data_generate(d, alpha, gamma_1, gamma_2, sig_1, sig_2, Delta, vlaw):
    k1, k2, n = int(gamma_1*d), int(gamma_2*d), int(alpha*d**2)
    if vlaw == 'ones':
        v = tf.ones((k2,), dtype=tf.float32)
    if vlaw == 'gaussd':
        v = tf.reduce_mean(tf.sort(tf.random.normal((100, k2))), axis=0)
    if vlaw == 'radd':
        l = int(k2/2)
        v = tf.concat([tf.ones(l, dtype=tf.float32), -tf.ones(k2-l, dtype=tf.float32)], axis=0)
    W1 = tf.random.normal((k1, d), dtype=tf.float32)
    W2 = tf.random.normal((k2, k1), dtype=tf.float32)
    X = tf.random.normal((d, n), dtype=tf.float32)
    Z = tf.random.normal((n,), dtype=tf.float32)
    Y = pre_activation(v, sig_2, W2, sig_1, W1, X) + np.sqrt(Delta) * Z
    return W1, W2, v, X, Y

def H(W1, W2, X, Y, v, Delta, sig_1, sig_2):
    D = pre_activation(v, sig_2, W2, sig_1, W1, X) - Y
    H1 = -tf.reduce_sum(D**2)/(2*Delta)
    H0 = -tf.reduce_sum(W1**2)/2 - tf.reduce_sum(W2**2)/2
    return H0 + H1

def H_vec(W_vec, X, Y, v, Delta, sig_1, sig_2, k2, k1, d):
    W1 = tf.reshape(W_vec[:k1*d], (k1, d))
    W2 = tf.reshape(W_vec[k1*d:], (k2, k1))
    return H(W1, W2, X, Y, v, Delta, sig_1, sig_2)

def hmc(params, W1_init, W2_init, v, X, Y, Delta, sig_1, sig_2):
    k1, d = W1_init.shape
    k2, _ = W2_init.shape
    step_size = params['step_size']
    num_leapfrog_steps = params['num_leapfrog_steps']
    num_adaptation_steps = params['num_adaptation_steps']

    W_init = tf.concat([tf.reshape(W1_init, [-1]), tf.reshape(W2_init, [-1])], axis=0)
    
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda W_vec: H_vec(W_vec, X, Y, v, Delta, sig_1, sig_2, k2, k1, d),
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps)
    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=hmc_kernel,
        num_adaptation_steps=num_adaptation_steps)
    @tf.function
    def run_chain():
        return tfp.mcmc.sample_chain(
            num_results=num_adaptation_steps,
            current_state=W_init,
            kernel=adaptive_kernel,
            trace_fn=None)
    Ws = run_chain()
    W1s = tf.reshape(Ws[:, :k1*d], (Ws.shape[0], k1, d))
    W2s = tf.reshape(Ws[:, k1*d:], (Ws.shape[0], k2, k1))
    return W1s, W2s

def test_error(W1s, W2s, W10, W20, v, sig_1, sig_2, Xtest):
    Ytest = pre_activation(v, sig_2, W20, sig_1, W10, Xtest)
    errors = []
    for i in range(W1s.shape[0]):
        Yhat = pre_activation(v, sig_2, W2s[i], sig_1, W1s[i], Xtest)
        error = tf.reduce_mean((Yhat - Ytest)**2)/2
        errors.append(error)
    return errors
