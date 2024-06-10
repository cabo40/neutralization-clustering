import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull

S1 = (0.05, 0.001)
S2 = (0, 0.005)
S3 = (0.1, 0.01)
S = [S1, S2, S3]
muS = [Si[0] for Si in S]
rho = [[1, 0.8, 0.4],
       [0.8, 1, 0.6],
       [0.4, 0.6, 1]]
rho = np.array(rho)
sigmaS = np.sqrt([Si[1] for Si in S])
sigmaS = sigmaS[:,None] * rho * sigmaS


rng = np.random.default_rng(0)
N = 1_000
w = rng.dirichlet([1, 1, 1], N)

mu = [x @ muS for x in w]
sigma = [np.sqrt(x @ sigmaS @ x) for x in w]

fig, ax = plt.subplots()
ax.scatter(sigma, mu)
hull = ConvexHull(list(zip(sigma, mu)))
line_segments = [hull.points[simplex] for simplex in hull.simplices]
line_segments = [x for x in line_segments if np.min([y[1] for y in x]) >= 0.05]
ax.add_collection(LineCollection(line_segments,
                                 colors='r',
                                 linestyle='solid',
                                 linewidth=2))
plt.xlabel("Standard Deviation")
plt.ylabel("Expected Returns")
plt.xlim([0, 0.12])
plt.ylim([-.01, 0.11])
plt.tight_layout()
plt.savefig("../contents/img/generated/Markowitz_bullet.pdf")
plt.show()


from pypfopt import EfficientFrontier
risk_free = 0.025
ef = EfficientFrontier(muS, sigmaS)
w_sharpe = ef.max_sharpe(risk_free_rate=risk_free)
w_sharpe = ef.clean_weights()
w_sharpe = np.array(list(w_sharpe.values()))
eff_sigma = np.sqrt(w_sharpe @ sigmaS @ w_sharpe)
eff_mu = muS @ w_sharpe
x = np.linspace(0, 2.5, 100)
cml = np.array([0, risk_free]) * (1-x[:,None]) + np.array([eff_sigma, eff_mu]) * x[:,None]


fig, ax = plt.subplots()
ax.scatter(sigma, mu)
ax.add_collection(LineCollection(line_segments,
                                 colors='r',
                                 linestyle='solid',
                                 linewidth=2))
plt.xlabel("Standard Deviation")
plt.ylabel("Expected Returns")
# plt.plot(0, risk_free, 'ro', c='green')
plt.plot(eff_sigma, eff_mu, 'ro', c='orange', zorder=999)
plt.plot(cml[:,0], cml[:,1], c='green')
plt.xlim([0, 0.12])
plt.ylim([-.01, 0.11])
plt.tight_layout()
plt.savefig("../contents/img/generated/Markowitz_bullet_rf.pdf")
plt.show()
