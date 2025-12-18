import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def pre_activation(v, sig, W, X):
    k, d = W.shape
    c_d, c_k = 1/np.sqrt(d), 1/np.sqrt(k)
    M = W @ X * c_d
    M = sig(M)
    return tf.tensordot(v, M, axes=1) * c_k

def data_generate(d, alpha, gamma, sig, Delta, vlaw, X):
    k, n = int(gamma*d), int(alpha*d**2)
    if vlaw == 'ones':
        v = tf.ones((k,), dtype=tf.float32)
    if vlaw == 'gaussd':
        v = tf.reduce_mean(tf.sort(tf.random.normal((10000, k))), axis=0)
        v = v * np.sqrt(k) / tf.norm(v)
    if vlaw == 'radd':
        l = int(k/2)
        v = tf.concat([tf.ones(l, dtype=tf.float32), -tf.ones(k-l, dtype=tf.float32)], axis=0)
    W = tf.random.normal((k, d), dtype=tf.float32)
    Z = tf.random.normal((n,), dtype=tf.float32)
    Y = pre_activation(v, sig, W, X) + np.sqrt(Delta) * Z
    return W, v, Y

def H(W, X, Y, v, Delta, sig):
    D = pre_activation(v, sig, W, X) - Y
    H1 = -tf.reduce_sum(D**2)/(2*Delta)
    H0 = -tf.reduce_sum(W**2)/2
    return H0 + H1

def H_vec(W_vec, X, Y, v, Delta, sig, k, d):
    W = tf.reshape(W_vec, (k, d))
    return H(W, X, Y, v, Delta, sig)

def hmc(params, W0, v, X, Y, Delta, sig, info):
    k, d = W0.shape
    step_size = params['step_size']
    num_leapfrog_steps = params['num_leapfrog_steps']
    num_adaptation_steps = params['num_adaptation_steps']
    if info:
        W_init = tf.reshape(W0, [-1])
    else:
        W_init = tf.random.normal((k*d,), dtype=tf.float32)
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda W_vec: H_vec(W_vec, X, Y, v, Delta, sig, k, d),
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
    Ws = tf.reshape(Ws, (Ws.shape[0], k, d))
    return Ws

def test_error(Ws, W0, v, sig, Xtest):
    Ytest = pre_activation(v, sig, W0, Xtest)
    errors = []
    for i in range(Ws.shape[0]):
        Yhat = pre_activation(v, sig, Ws[i], Xtest)
        error = tf.reduce_mean((Yhat - Ytest)**2)/2
        errors.append(error)
    return errors
