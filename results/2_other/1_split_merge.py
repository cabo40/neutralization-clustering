import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

fig, axs = plt.subplots(3, 1, figsize=(5.5, 2.5), layout="constrained",
                        sharex=True)
rng = np.random.default_rng(0)
X = np.linspace(-12, 12, 1000)
fy = norm.pdf(X+5)*0.5+norm.pdf(X)*0.25+norm.pdf(X-4)*0.25
axs[0].plot(X, fy, label='True density')
axs[0].legend()
fy = norm.pdf(X, loc=-5.1, scale=0.98)*0.49+norm.pdf(X, loc=1.9, scale=2.63)*0.51
axs[1].plot(X, fy, label='Scenario A density')
axs[1].legend()
fy = norm.pdf(X,loc=-5.75, scale=0.5)*0.25+norm.pdf(X, loc=-4.25, scale=0.5)*.25+norm.pdf(X)*0.25+norm.pdf(X-4)*0.25
axs[2].plot(X, fy, label='Scenario B density')
axs[2].legend()
fig.supylabel('Density')
axs[2].set_xlabel('x')
plt.savefig('figs/split_merge_dens.pdf')
plt.show()
