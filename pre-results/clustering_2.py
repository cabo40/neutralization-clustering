import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def dist(a, d):
    return sum([d(a, x) for x in X])


x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([-0.3, 0.6])
X = np.array([x, y, z])
ul1 = opt.minimize(dist, np.array([1, 1]),
                   args=(lambda x, y: np.linalg.norm(x - y, ord=1),)).x
ul2 = opt.minimize(dist, np.array([1, 1]),
                   args=(lambda x, y: np.linalg.norm(x - y, ord=2),)).x
uv = opt.minimize(dist, np.array([1, 1]),
                 args=(lambda x, y: np.linalg.norm(x - y, ord=2)**2,)).x
ulinf = opt.minimize(dist, np.array([1, 1]),
                 args=(lambda x, y: np.linalg.norm(x - y, ord=np.inf),)).x

fig, axes = plt.subplots(1,2, figsize=np.array([6.4, 2.4])*0.9, sharey=True)
axes[0].set_ylim([-0.1, 1.1])
# axes[0].set_xlabel("x")
# axes[0].set_ylabel("y")
axes[0].scatter(X[:, 0], X[:, 1], c='r')
axes[0].scatter(ul1[0], ul1[1], c='g', marker='^', s=80)
axes[0].text(ul1[0]+.05, ul1[1]-0.05, '$L_1$')
axes[0].scatter(ul2[0], ul2[1], c='g', marker='s', s=80)
axes[0].text(ul2[0]+.05, ul2[1]-0.05, '$L_2$')
axes[0].scatter(uv[0], uv[1], c='g', marker='p', s=80)
axes[0].text(uv[0]+.05, uv[1]-0.05, 'Var')
axes[0].scatter(ulinf[0], ulinf[1], c='g', marker='o', s=80)
axes[0].text(ulinf[0]+.05, ulinf[1]-0.05, r'$L_\infty$')

z = np.array([0, 0])
X = np.array([x, y, z])
ul1 = opt.minimize(dist, np.array([1, 1]),
                   args=(lambda x, y: np.linalg.norm(x - y, ord=1),)).x
ul2 = opt.minimize(dist, np.array([1, 1]),
                   args=(lambda x, y: np.linalg.norm(x - y, ord=2),)).x
uv = opt.minimize(dist, np.array([1, 1]),
                 args=(lambda x, y: np.linalg.norm(x - y, ord=2)**2,)).x
ulinf = opt.minimize(dist, np.array([1, 1]),
                 args=(lambda x, y: np.linalg.norm(x - y, ord=np.inf),)).x
# axes[1].set_xlabel("x")
axes[1].scatter(ul1[1], ul1[1], c='g', marker='^', s=80)
axes[1].text(ul1[1]+.05, ul1[1]-0.05, '$L_1$')
axes[1].scatter(ul2[1], ul2[1], c='g', marker='s', s=80)
axes[1].text(ul2[1]+.05, ul2[1]-0.05, '$L_2$')
axes[1].scatter(uv[1], uv[1], c='g', marker='p', s=80)
axes[1].text(uv[1]+.05, uv[1]-0.05, 'Var')
axes[1].scatter(ulinf[1], ulinf[1], c='g', marker='o', s=80)
axes[1].text(ulinf[1]+.05, ulinf[1]-0.05, r'$L_\infty$')
axes[1].scatter(X[:, 0], X[:, 1], c='r')
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[1].set_xlabel("x")
fig.tight_layout()
plt.savefig("../contents/img/generated/cluster_2.pdf")
plt.show()
