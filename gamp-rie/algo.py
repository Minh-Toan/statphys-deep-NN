import numpy as np
from scipy.integrate import quad

def Ez(f):
    integrand = lambda z: f(z)*np.exp(-z**2/2)/np.sqrt(2*np.pi)
    return quad(integrand, -10, 10)[0]

def coeffs(f):
    c0 = Ez(lambda z: f(z))
    c1 = Ez(lambda z: z*f(z))
    c2 = Ez(lambda z: (z**2-1)*f(z))/np.sqrt(2)
    nu = Ez(lambda z: f(z)**2)
    return c0, c1, c2, nu

def v_generate(k, vlaw):
    if vlaw=='ones':
        return np.ones(k)
    if vlaw=='rad':
        l = int(k/2)
        return np.array([1]*l + [-1]*(k-l))
    if vlaw=='4point':
        l = int(k/4)
        v3_ = -3*np.sqrt(1/5)*np.ones(l)
        v1_ = -np.sqrt(1/5)*np.ones(l)
        v1 = np.sqrt(1/5)*np.ones(l)
        v3 = 3*np.sqrt(1/5)*np.ones(k-3*l)        
        v =  np.concatenate([v3_, v1_, v1, v3])
        return v*np.sqrt(k)/np.linalg.norm(v)
    if vlaw=='unif':
        v = np.linspace(-1,1,k)
        return v*np.sqrt(k)/np.linalg.norm(v)
    if vlaw=='gauss':
        v = np.mean(np.sort(np.random.randn(10000, k)), axis=0)
        return v*np.sqrt(k)/np.linalg.norm(v)

def data_generate(d, gamma, alpha, Delta, sig, vlaw, prior):
    k = int(gamma*d)
    n = int(alpha*d**2)
    v = v_generate(k, vlaw)
    if prior=='rad':
        W0 = np.random.choice([-1,1],(k,d))
    if prior=='gauss':
        W0 = np.random.randn(k,d)
    X = np.random.normal(size=(d,n))
    Z = np.random.randn(n)
    U0 = W0@X/np.sqrt(d)
    Y = sig(U0).T@v/np.sqrt(k) + np.sqrt(Delta)*Z
    return W0, X, Y, v

def AMP(X, y, gamma, noise, f_RIE, F_RIE, iterations, damping, tol):
    
    def g_out(y, w, V):
        return (y-w)/(noise + V)

    d, n = X.shape
    k = int(gamma*d)
    alpha = n/d**2
        
    W = np.random.randn(k, d)
    hatS = W.T@W/k
        
    hatC    = 10.
    omega   = np.ones(n)*10.
    V       = 10.

    error = np.inf
    for t in range(iterations):
        newV = hatC
        SX = hatS@X 
        newOmega = np.sum(X*SX, axis=0)/np.sqrt(d) - np.trace(hatS)/np.sqrt(d) - g_out(y, omega, V)*newV
        
        V = newV * (1-damping) + V * damping
        omega = newOmega * (1-damping) + omega * damping

        gs = g_out(y, omega, V)
        A_normalised = (2*alpha/n)*np.sum(gs**2)
        R = hatS + 1/(A_normalised*d**(3/2))*( X@(X*gs[None,:]).T - np.sum(gs)*np.eye(d) )
        
        noise_A = 1/(2*A_normalised)
        new_hatS = f_RIE(R, noise_A)
        hatC = 2*F_RIE(noise_A)
        
        error = np.linalg.norm(hatS - new_hatS)**2/d
        hatS = new_hatS
        
        if error < tol:
            break

    return np.sqrt(k)*hatS

def algo_perf(d, gamma, alpha, Delta, f_RIE, F_RIE, sig, vlaw, prior):
    '''
    generate one dataset, perform the algorithm on this dataset, and compute the generalization error
    '''
    k = int(gamma*d)
    n = int(alpha*d**2)
    W0, X, Y, v = data_generate(d, gamma, alpha, Delta, sig, vlaw, prior)

    c0, c1, c2, nu = coeffs(sig)
    
    Y0 = c0*np.sum(v)/np.sqrt(k)

    # estimate S1 = v.T W/ sqrt(k) 
    if abs(c1**2/nu) > 1e-3:
        Delta_ = Delta + nu - c0**2 - c1**2
        Y_  = (Y-Y0)/np.sqrt(Delta_)
        X_ = c1*X/np.sqrt(d*Delta_)
        S1_hat = np.linalg.solve(np.eye(d) + (X_@X_.T), X_@Y_)
    else:
        S1_hat = np.zeros(d)
    Y1 = c1*S1_hat@X/np.sqrt(d)
    
    # estimate S2 = W.T Dv W / sqrt(k)
    if abs(c2**2/nu) > 1e-3:
        Y_ = np.sqrt(2/gamma)*(Y - Y0 -Y1)/c2 # we're denoising W.T W/k
        noise = 2*(Delta+nu-c0**2-c1**2-c2**2)/(gamma*c2**2)
        S2_hat = AMP(X, Y_, gamma, noise, f_RIE, F_RIE, iterations = 100, damping = 0.4, tol = 1e-4)
    else:
        S2_hat = np.zeros((d,d))

    # test error of the algorithm
    ntest = 10**4
    errs = []
    for i in range(ntest):
        xtest = np.random.randn(d)
        ytest = v.T@sig(W0@xtest/np.sqrt(d))/np.sqrt(k)
        Ztest = np.outer(xtest, xtest)-np.identity(d)
        yhat  = c0*np.sum(v)/np.sqrt(k) + c1*S1_hat@xtest/np.sqrt(d) + c2*np.sum(Ztest*S2_hat)/(d*np.sqrt(2))
        err = (ytest-yhat)**2
        errs.append(err)
    return np.mean(errs)

    