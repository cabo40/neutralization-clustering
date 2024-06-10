import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

X = np.concatenate([
    rng.multivariate_normal([-2.5, -2.5], np.identity(2), size=50),
    rng.multivariate_normal([2.5, 2.5], np.identity(2), size=50)])
X = np.array(sorted(X, key=lambda x: x[0]))

fig, axes = plt.subplots(1,2, figsize=[6.4, 2.4])
axes[0].scatter(X[:, 0], X[:, 1], s=25, c='black')
l = 5
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_xlim([-l, l])
axes[0].set_ylim([-l, l])

axes[1].scatter(X[:50, 0], X[:50, 1], s=25, c='green', marker="^")
axes[1].scatter(X[50:, 0], X[50:, 1], s=25, c='blue', marker="s")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_xlim([-l, l])
axes[1].set_ylim([-l, l])
fig.tight_layout()
plt.savefig("../contents/img/generated/cluster_1.pdf")
plt.show()
