g, g_prime, coeffs = func.gg(sig_n)
qs = np.linspace(0, 1, 201)
gs = [func.h(q, sig_n) for q in qs]
plt.plot(qs, gs)
plt.plot(qs, g(qs))
plt.show()


def diff(f,x): # f defined on [0, 1]
    e1, e2 = 1e-6, 1e-10
    if x<e1:
        return (f(x+e2)-f(x))/e2
    if x>1-e1:
        return (f(x) - f(x-e2))/e2
    return (f(x+e1)-f(x-e1))/(2*e1)


'''
1D potential
'''
Delta = 0.1
alpha = 1.16
qs = np.linspace(0, 0.99, 101).reshape(-1, 1)
pots = [potential(alpha, Delta, q) for q in qs]
plt.plot(qs, pots)

'''
2D potential
'''
alpha = 1
Delta = 0.1

p = lambda q: potential(alpha, Delta, q)

x = np.linspace(0, 0.2, 21)
y = np.linspace(0, 0.2, 21)
X, Y = np.meshgrid(x, y)
Z = p(np.array([X, Y]))
plt.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='coolwarm')
plt.colorbar()
plt.show()


'''
old generalisation error
'''
cmap = plt.get_cmap("hot")
colors = [cmap(i) for i in np.linspace(0, 0.6, 5)]

width, height = 7, 6
size = int(width*3)
fig, ax = plt.subplots(figsize=(width, height), tight_layout=True)

ax.tick_params(axis='both', labelsize=size)
ax.tick_params(axis='both', labelsize=size)
ax.set_xlabel(r'$\alpha$', fontsize=int(1.2*size))
ax.set_ylabel(r'$\varepsilon^{\rm opt}$', fontsize=int(1.2*size))
ax.set_xlim(1, 18)
ax.set_xticks([0, 5, 10, 15])
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
for L in range(1, 6):
    results = np.loadtxt(f'L_{L}.csv', delimiter=',')
    plt.plot(alphas, results[0], color=colors[L-1], label=fr'$L={L}$')
plt.legend(fontsize=size)
plt.grid()
plt.savefig('deep-tanh-error-1.pdf')


