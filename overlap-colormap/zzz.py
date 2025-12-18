

S = np.zeros((k2, k1))
for i in range(nsample):
    W2a = np.loadtxt(f'alpha_{alpha}/W2a_{i+1}.csv', delimiter=',')
    W20 = np.loadtxt(f'alpha_{alpha}/W20_{i+1}.csv', delimiter=',')
    v2 = v.T@W2a/np.sqrt(k2)
    W2a = W2a[np.argsort(v)][:, np.argsort(v2)]
    W20 = W20[np.argsort(v)][:, np.argsort(v2)]
    v2=np.sort(v2)
    S += W2a*W20

S = S/nsample

m = 4
S_ = S.reshape(int(k2/m), m, int(k1/m), m).mean(axis=(1, 3))
plt.imshow(np.abs(S_), origin='lower')
plt.xticks([])
plt.yticks([])
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=15)
plt.savefig('Q2heatmap_numerical_0.pdf')
plt.show()



def bin_grid(W, v, v2, a, b, m):
    G = np.linspace(a, b, m + 1)
    i = np.digitize(v, G) - 1
    j = np.digitize(v2, G) - 1
    mask_i = (i >= 0) & (i < m)
    mask_j = (j >= 0) & (j < m)
    valid_i, valid_j = np.nonzero(np.outer(mask_i, mask_j))
    ii = i[valid_i]
    jj = j[valid_j]
    W_flat = W[np.ix_(mask_i, mask_j)].ravel()
    Wsum = np.zeros((m, m))
    Wcount = np.zeros((m, m))
    np.add.at(Wsum, (ii, jj), W_flat)
    np.add.at(Wcount, (ii, jj), 1)
    return Wsum, Wcount

'''
old colormap
'''
size=15
fig, ax = plt.subplots(tight_layout=True)
im = ax.imshow(np.abs(result), origin='lower', extent=[vmin, vmax, vmin, vmax])
ax.set_xticks(np.linspace(vmin, vmax, 5))
ax.set_yticks(np.linspace(vmin, vmax, 5))
ax.tick_params(axis='x', labelsize=size)
ax.tick_params(axis='y', labelsize=size)
ax.set_xlabel(r'$v^{(2)}$', fontsize=int(1.2*size))
ax.set_ylabel(r'$v$', fontsize=int(1.2*size))

cbar = fig.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=size)
plt.title(fr'$\alpha={alpha}$', fontsize=size)
plt.savefig(f'Q2heatmap_numerical_{alpha}.pdf')
# plt.show()