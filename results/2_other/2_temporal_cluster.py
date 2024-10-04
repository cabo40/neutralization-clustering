import matplotlib.pyplot as plt
import numpy as np
colors = np.array([plt.cm.Set1(i) for i in range(3)])
fig, axs = plt.subplots(1, 1, figsize=(5.5, 2), layout="constrained",
                        sharex=True)
ax = axs
ax.scatter(np.arange(5)+1, np.repeat(1, 5), s=500,
           c=colors[[0,0,0,1,1]], cmap="Accent")
ax.scatter(np.arange(5)+1, np.repeat(2, 5), s=500, marker="^",
           c=colors[[0,1,2,2,2]], cmap="Accent")
ax.scatter(np.arange(5)+1, np.repeat(3, 5), s=500, marker="s",
           c=colors[[1,2,1,0,0]], cmap="Accent")
ax.set_xlim((0.5,5.5))
ax.set_ylim((0.1,3.5))
ax.set_xlabel("Time")
ax.set_ylabel("Index")
ax.spines[['right', 'top']].set_visible(False)
plt.savefig('figs/temporal_cluster.pdf')
plt.show()