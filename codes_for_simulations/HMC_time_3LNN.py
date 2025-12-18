import os
import csv
import jax
import scipy.integrate
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np
import math
from jax import random

# -----------------------------
# Configuration
# -----------------------------
jax.config.update("jax_enable_x64", False)

d_vals = [50, 100, 150, 200, 250, 300, 350]  # input dimensions
gamma1 = 0.5            # k1/d
gamma2 = 0.5            # k2/d  
alpha = 4.0             # n/d^2
delta = 0.2             # noise variance
sigma_const = math.sqrt(delta)
info_init = 0           # whether to use informative initialization
quenched = 1            # whether to use quenched disorder for v
lr = 0                  # learn readout v if lr=1; else fix to teacher v

num_warmup_samples = 2000                   # number of warmup samples to collect (with trajectory)
num_posterior_samples = 200
points = 300            

tree_depth = 8          # NUTS tree depth
n_test = 10000          # test set size

num_teachers = 10       # run 10 teacher instances per dimension


var_tanh, error = scipy.integrate.quad(lambda x: (np.tanh(2*x))**2 *np.exp(-x**2/2)/(np.sqrt(2*np.pi)), -10, 10)

def activation(x):
    return (jax.nn.tanh(2*x))/np.sqrt(var_tanh)  

def make_teacher_forward(W1_teacher, W2_teacher, v_teacher, d, k1, k2):
    """Return a function that maps X -> y using the teacher two-layer network."""
    def teacher_forward(x):
        h1 = activation((1.0 / jnp.sqrt(d)) * x @ W1_teacher.T)          # (n, k1)
        h2 = activation((1.0 / jnp.sqrt(k1)) * h1 @ W2_teacher.T)       # (n, k2)
        return (1.0 / jnp.sqrt(k2)) * h2 @ v_teacher                    # (n,)
    return teacher_forward

def student_predict(W1, W2, X, d, k1, k2, v):
    """Predict with a two-hidden-layer set of weights W1 (k1,d), W2 (k2,k1) and readout v (k2,)"""
    h1 = activation((1.0 / jnp.sqrt(d)) * X @ W1.T)
    h2 = activation((1.0 / jnp.sqrt(k1)) * h1 @ W2.T)
    return (1.0 / jnp.sqrt(k2)) * h2 @ v

def read_done_teachers_from_csv(csv_name, current_d, gamma1, gamma2):
    done = set()
    if not os.path.exists(csv_name):
        return done
    try:
        with open(csv_name, newline='') as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                try:
                    d_val_s = row.get('d', '')
                    t_idx_s = row.get('teacher_idx', row.get('teacher_id', ''))
                    if d_val_s is None or d_val_s == '':
                        raise ValueError("missing 'd'")
                    if t_idx_s is None or t_idx_s == '':
                        raise ValueError("missing 'teacher_idx/teacher_id'")

                    d_val = int(float(d_val_s))
                    t_idx = int(float(t_idx_s))
                except Exception as e:
                    print(f"Skipping existing CSV row {row_idx} due to missing d or teacher_idx: {e}")
                    continue

                if d_val != current_d:
                    continue

                k1_row = int(gamma1 * d_val)
                k2_row = int(gamma2 * d_val)
                if k1_row <= 0 or k2_row <= 0 or d_val <= 0:
                    print(f"Skipping existing CSV row {row_idx} with non-positive dimensions d={d_val}, k1={k1_row}, k2={k2_row}")
                    continue

                done.add(int(t_idx))
    except Exception as e:
        print(f"Could not read existing CSV {csv_name}: {e}. Will re-run all teachers for d={current_d}.")
        done = set()
    return done

def append_rows_to_csv(csv_name, rows, fieldnames):
    """Append rows (list of dicts) to csv_name. If file does not exist, create and write header."""
    write_header = not os.path.exists(csv_name)
    with open(csv_name, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

# -----------------------------
# Main loop over dimensions
# -----------------------------
master_seed = 2
master_key = jax.random.PRNGKey(master_seed)

for idx, d in enumerate(d_vals):
    print(f"\n--- Running for d = {d} ({idx+1}/{len(d_vals)}) ---")

    k1 = int(gamma1 * d)
    k2 = int(gamma2 * d)
    n = int(alpha * d**2)
    sigma = sigma_const

    # CSV filename for this dimension
    csv_name = f'gibbs_error_3LNN_ones_tanh_d{d}.csv'

    # Determine which teacher indices are already present in the CSV (if any)
    done_teachers = read_done_teachers_from_csv(csv_name, d, gamma1, gamma2)
    if done_teachers:
        print(f"Found existing entries for d={d}, skipping teacher indices: {sorted(done_teachers)}")

    rows_to_append = []

    for t in range(num_teachers):
        if t in done_teachers:
            print(f"[d={d}] Teacher {t} already present in {csv_name}, skipping.")
            continue

        # split fresh RNGs for this teacher instance
        master_key, key_teacher, key_data, key_test = jax.random.split(master_key, 4)

        key_w1, key_w2, key_v = jax.random.split(key_teacher, 3)
        W1_teacher = jax.random.normal(key_w1, shape=(k1, d)).astype(jnp.float32)
        W2_teacher = jax.random.normal(key_w2, shape=(k2, k1)).astype(jnp.float32)
        v_teacher = jnp.ones((k2,), dtype=jnp.float32)

        # generate training data (use independent keys for X and noise)
        key_X, key_noise, key_init = jax.random.split(key_data, 3)
        X = jax.random.normal(key_X, (n, d), dtype=jnp.float32)
        teacher_forward_data = make_teacher_forward(W1_teacher, W2_teacher, v_teacher, d, k1, k2)
        y = teacher_forward_data(X) + sigma * jax.random.normal(key_noise, (n,), dtype=jnp.float32)

        # probabilistic model (sample W1 and W2; sample v only if lr is truthy)
        if lr:
            def model(X, y=None):
                W1 = numpyro.sample("W1", dist.Normal(0, 1).expand([k1, d]).to_event(2))
                W2 = numpyro.sample("W2", dist.Normal(0, 1).expand([k2, k1]).to_event(2))
                if quenched:
                    v = numpyro.sample("v", dist.Normal(v_teacher, 0.0001).expand([k2]).to_event(1))
                else:
                    v = numpyro.sample("v", dist.Normal(0, 1).expand([k2]).to_event(1))
                h1 = activation((1.0 / jnp.sqrt(d)) * X @ W1.T)
                h2 = activation((1.0 / jnp.sqrt(k1)) * h1 @ W2.T)
                f = (1.0 / jnp.sqrt(k2)) * h2 @ v
                with numpyro.plate("obs", X.shape[0]):
                    numpyro.sample("y", dist.Normal(f, sigma), obs=y)
        else:
            def model(X, y=None):
                W1 = numpyro.sample("W1", dist.Normal(0, 1).expand([k1, d]).to_event(2))
                W2 = numpyro.sample("W2", dist.Normal(0, 1).expand([k2, k1]).to_event(2))
                h1 = activation((1.0 / jnp.sqrt(d)) * X @ W1.T)
                h2 = activation((1.0 / jnp.sqrt(k1)) * h1 @ W2.T)
                f = (1.0 / jnp.sqrt(k2)) * h2 @ v_teacher
                with numpyro.plate("obs", X.shape[0]):
                    numpyro.sample("y", dist.Normal(f, sigma), obs=y)

        # init_params
        if lr:
            if info_init:
                init_params = {"W1": W1_teacher, "W2": W2_teacher, "v": v_teacher}
            else:
                init_params = {"W1": jax.random.normal(key_init, (k1, d), dtype=jnp.float32),
                               "W2": jax.random.normal(key_init, (k2, k1), dtype=jnp.float32),
                               "v": v_teacher}
        else:
            if info_init:
                init_params = {"W1": W1_teacher, "W2": W2_teacher}
            else:
                init_params = {"W1": jax.random.normal(key_init, (k1, d), dtype=jnp.float32),
                               "W2": jax.random.normal(key_init, (k2, k1), dtype=jnp.float32)}

        # run NUTS warmup and collect warmup
        nuts_kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.65,
                           dense_mass=False, max_tree_depth=tree_depth)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup_samples, num_samples=num_posterior_samples)

        print(f"[d={d}] Teacher {t+1}/{num_teachers}: Starting warmup (collecting warmup trajectory)...")
        mcmc.warmup(master_key, X, y=y, init_params=init_params, collect_warmup=True)

        samples = mcmc.get_samples()
        # Expect both W1 and W2 in samples
        if 'W1' not in samples or 'W2' not in samples:
            raise RuntimeError("W1 or W2 not found in MCMC samples. Check NumPyro version or warmup collection.")

        samples_W1 = samples['W1']  # shape: (num_samples, k1, d)
        samples_W2 = samples['W2']  # shape: (num_samples, k2, k1)

        # test set 
        teacher_forward = make_teacher_forward(W1_teacher, W2_teacher, v_teacher, d, k1, k2)
        X_test = jax.random.normal(key_test, (n_test, d))
        y_test = teacher_forward(X_test)

        gibbs_error = []
        iterations = []

        num_samples_available = samples_W1.shape[0]
        step = max(1, int(num_samples_available / points))

        for i in range(0, num_samples_available, step):
            W1_sample = samples_W1[i]
            W2_sample = samples_W2[i]
            v_sample = v_teacher if not lr else (samples.get('v', v_teacher)[i] if 'v' in samples else v_teacher)

            iter_val = int(tree_depth * (i + 1))
            iterations.append(iter_val)

            # predictions on test (gibbs)
            pred_test = student_predict(W1_sample, W2_sample, X_test, d, k1, k2, v_sample)
            y_test_np = np.asarray(y_test)
            pred_test_np = np.asarray(pred_test)
            g_err = float(np.mean((y_test_np - pred_test_np)**2 / 2.0))

            gibbs_error.append(g_err)

        for it_val, g_err in zip(iterations, gibbs_error):
            rows_to_append.append({
                'd': str(int(d)),
                'iteration': str(int(it_val)),
                'teacher_idx': str(int(t)),
                'gibbs_error': repr(float(g_err))
            })

        # print the last error computed for this teacher instance
        if gibbs_error:
            print(f"[d={d}] Teacher {t+1}/{num_teachers}: last warmup test error = {gibbs_error[-1]:.6e}")
        else:
            print(f"[d={d}] Teacher {t+1}/{num_teachers}: (warning) no warmup test errors collected")

    if rows_to_append:
        fieldnames = ['d', 'iteration', 'teacher_idx', 'gibbs_error']
        append_rows_to_csv(csv_name, rows_to_append, fieldnames)
        print(f"Appended {len(rows_to_append)} new rows to {csv_name}.")
    else:
        print(f"No new teacher runs for d={d}; {csv_name} unchanged.")

