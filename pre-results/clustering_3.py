import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

dist1 = norm(loc=-4, scale=2)
dist2 = norm(loc=1, scale=1)
dist3 = norm(loc=5, scale=1.5)

x = np.linspace(-10, 10, 1000)

d1 = dist1.pdf(x)
d2 = dist2.pdf(x)
d3 = dist3.pdf(x)
ds = d1 + d2 + d3
x_w_pre = np.where(np.diff(ds) < 0)[0][0]
x_w = np.where(np.diff(ds[x_w_pre + 1:]) > 0)[0][0] + x_w_pre
x_w_pre_2 = np.where(np.diff(ds[x_w + 1:]) < 0)[0][0] + x_w
x_w_2 = np.where(np.diff(ds[x_w_pre_2 + 1:]) > 0)[0][0] + x_w_pre_2

fig, ax = plt.subplots(figsize=[6.4, 3.4])
ax.plot(x, ds, label="Generating density")
ax.fill_between(
    x[:x_w], np.zeros(x_w), ds[:x_w], label="Region 1"
)
ax.fill_between(
    x[x_w:],
    np.zeros(len(x[x_w:])),
    ds[x_w:],
    label="Region 2",
)
ax.fill_between(
    x[x_w_2:],
    np.zeros(len(x[x_w_2:])),
    ds[x_w_2:],
    label="Region 3",
)

ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title("Density based clustering")
ax.legend()
fig.tight_layout()
plt.savefig("../contents/img/generated/cluster_3.pdf")
plt.show()
