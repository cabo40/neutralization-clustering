import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

dist1 = norm(loc=0, scale=1)
dist2 = norm(loc=0, scale=5)

x = np.linspace(-10, 10, 1000)

d1 = dist1.pdf(x) * 0.5
d2 = dist2.pdf(x) * 0.5
ds = d1 + d2
x_w = np.where(d1 > d2)[0][0]
x_w_2 = np.where(d1[x_w:] < d2[x_w:])[0][0] + x_w

fig, ax = plt.subplots(figsize=[6.4, 3.4])
ax.plot(x, d1, label="Component 1", color=(0,0,1,0.5), linestyle=':')
ax.plot(x, d2, label="Component 2", color=(0,0,1,0.5), linestyle='-.')
ax.plot(x, ds, label="Mixture", color="green")
ax.axvline(x[x_w], color="grey", linestyle="--", label="Region boundary")
ax.axvline(x[x_w_2], color="grey", linestyle="--")
ax.fill_between(
    x,
    d1,
    np.maximum(d1, d2),
    label="Region 1",
    color=(0,0,1,0.5)
)
ax.fill_between(
    x,
    d2,
    np.maximum(d1, d2),
    label="Region 2",
    color=(0,0.5,0.5,0.5)
)


ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title("Component based clustering")
ax.legend()
fig.tight_layout()
plt.savefig("../contents/img/generated/local_minima_2.pdf")
plt.show()
