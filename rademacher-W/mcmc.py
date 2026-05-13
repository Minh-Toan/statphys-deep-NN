import numpy as np

def data_generate(d, gamma, alpha, Delta, v, sig):
    k = int(gamma*d)
    n = int(alpha*d**2)
    # 
    W0 = np.random.choice([-1,1],(k,d))
    
    X = np.random.normal(size=(d,n))
    Z = np.random.randn(n)
    U0 = W0@X/np.sqrt(d)
    Y = v.T@sig(U0)/np.sqrt(k) + np.sqrt(Delta)*Z
    return W0, X, Y

def metropolis(W0, X, Y, Delta, v, sig, ncycle, info):
    '''
    generate a sequence of W's by Markow Chain Monte Carlo
    '''
    k, d = W0.shape
    interval = int(ncycle*k*d/1000)
    # stored values during MCMC
    if info:
        W = W0.copy()
    else:
        W = np.random.choice([-1,1],(k,d))
    U = W@X/np.sqrt(d)
    D = sig(U).T@v/np.sqrt(k) - Y
    H = np.sum(D**2)/(2*Delta)
    Ws = []
    
    # MCMC
    for iteration in range(ncycle*k*d+1):
        s, t = np.random.randint(0,k), np.random.randint(0, d)
        dU = -2*W[s,t]*X[t]/np.sqrt(d)
        dD =  v[s]*(sig(U[s]+dU) - sig(U[s]))/np.sqrt(k)
        Hnew = np.sum((D+dD)**2)/(2*Delta)
        dH = Hnew - H
        if dH<0 or np.random.rand()<np.exp(-dH): # acceptance condition
            U[s] += dU
            D += dD
            H = Hnew
            W[s,t] *= -1
        if iteration%interval == 0:
            Ws.append(W.copy())     
    return Ws
    
         

    

