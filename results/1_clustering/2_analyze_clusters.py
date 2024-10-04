import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_pickle('crsp_small.pickle')
year = 2019
ret = df['RET'].loc[f'{year}-01-01':f'{year + 1}-01-01'].unstack()
grp = df['NAICS'].loc[f'{year}-01-01':f'{year + 1}-01-01'].unstack()
stable = grp.columns[grp.nunique(axis=0, dropna=False) == 1]
ret = ret.loc[:, stable]
grp = grp.loc[:, stable].iloc[0, :]
groups_i = pd.read_pickle(f'output_groups/{year}.pkl')
del groups_i['rand']
groups_i = groups_i.loc[ret.columns]

sigmo = ret.add(-ret.mean(axis=1), axis=0)
n = sigmo.shape[1]
sigmo = sigmo.iloc[:, :n]


def plot_sign(ord_rets, title, ax=None):
    X, Y = np.meshgrid(ord_rets.index, range(len(ord_rets.columns)))
    if ax is None:
        ax = plt.pcolor(X, Y, -ord_rets.T, cmap='bwr', vmin=-0.05, vmax=0.05)
    else:
        ax.pcolor(X, Y, -ord_rets.T, cmap='bwr', vmin=-0.04, vmax=0.04)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 4, 7, 10, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.yaxis.set_visible(False)
    ax.set_title(title)

    if ax is None:
        plt.xlabel("Time")


names = ['$k$-means', '$k$-medians', 'Agglomerative(E)', 'Agglomerative(mH)',
         'DBSCAN', 'Gaussian mixture', 'Dirichlet distribution',
         'Dirichlet Process', 'NAICS']
fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(7.5, 5.5), sharex=True,
                        layout="constrained")

for i, a in enumerate(groups_i.columns):
    j = int(i / 3)
    group = groups_i[a]
    group = np.argsort(np.unique(group, return_counts=True)[1])[
        np.unique(group, return_inverse=True)[1]]
    plot_sign(sigmo.iloc[:, np.argsort(group)], names[i], ax=axs[i % 3, j])
    axs[i % 3, j].tick_params(axis='x', labelrotation=45)
fig.suptitle('Log-returns visualization sorted by group')
plt.savefig('figs/groups_viz.png', dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(7.5, 3), layout="constrained")
rng = np.random.default_rng(1)
rand = list(range(sigmo.shape[1]))
rng.shuffle(rand)
plot_sign(sigmo.iloc[:, rand], 'Random log-returns visualization', ax=ax)
plt.savefig('figs/rand_viz.png')
plt.show()