import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
N = 100
noise_per_point = 10

x = []
y = []

# Spiral
for i in np.linspace(0, 1, N):
    angle = np.sqrt(i) * np.pi * 4
    r = np.sqrt(i)
    noise = rng.normal(loc=0, scale=0.02,
                       size=noise_per_point * 2).reshape(-1, 2)
    x.extend(r * np.cos(angle) + noise[:, 0] - 2.5)
    y.extend(r * np.sin(angle) + noise[:, 1])

# Triangle
for i in np.linspace(0, 1, N):
    noise = rng.normal(loc=0, scale=0.02,
                       size=noise_per_point * 2).reshape(-1, 2)
    if i < 1 / 3:
        x.extend(i * 9 + noise[:, 0] - 1.5)
        y.extend(noise[:, 1] - 0.8)
    elif i < 2 / 3:
        x.extend(9 * (i - 1 / 3) * np.cos(np.pi / 3) + noise[:, 0] - 1.5)
        y.extend(5 * (i - 1 / 3) * np.sin(np.pi / 3) + noise[:, 1] - 0.8)
    else:
        x.extend(-9 * (i - 2 / 3) * np.cos(np.pi / 3) + noise[:, 0] + 1.5)
        y.extend(5 * (i - 2 / 3) * np.sin(np.pi / 3) + noise[:, 1] - 0.8)

# Snake
for i in np.linspace(0, 1, N):
    noise = rng.normal(loc=0, scale=0.02,
                       size=noise_per_point * 2).reshape(-1, 2)
    x.extend(1.5 * (i - 0.5) + noise[:, 1] - 0.1 + 2.5)
    y.extend(np.cos(i * 4 * np.pi) * 0.7 + noise[:, 0] - 0.1)

X = np.array([x, y]).T

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.1)
c = dbscan.fit_predict(X)

fig, axes = plt.subplots(1, 2, figsize=np.array([6.4, 2.4]) * 0.9,
                         sharey=True, sharex=True)
axes[0].scatter(X[:, 0], X[:, 1], s=2, c=c)
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].set_title("DBSCAN")

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=0)
c = km.fit_predict(X)
axes[1].scatter(X[:, 0], X[:, 1], s=2, c=c)
axes[1].set_xlabel("X")
axes[1].set_title("$k$-means")
plt.tight_layout()
plt.savefig("../contents/img/generated/cluster_dbscan.pdf")
plt.show()
