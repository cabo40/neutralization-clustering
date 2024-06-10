from sklearn.metrics import mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import time
import cvxopt as opt
import cvxpy as cp
import pickle


# opt.solvers.options['maxiters'] = 5
# opt.solvers.options['abstol'] = 1e-5
# opt.solvers.options['reltol'] = 1e-2
# opt.solvers.options['feastol'] = 1e-6
def markowitz_port_old(r: pd.DataFrame):
    r_mu = r.mean(0).to_numpy()
    r_mu = np.append(r_mu, 1)
    r_s = r.cov().to_numpy()
    r_s = np.append(r_s, [np.ones(r_s.shape[1])], axis=0)
    w = scipy.optimize.lsq_linear(r_s, r_mu, (0, 1),
                                  method='trf', lsq_solver='lsmr',
                                  max_iter=20)["x"]
    w = w / sum(w)  # Allow for imperfections in fit, not in total allocation
    return pd.Series(w, index=r.columns)


def markowitz_port_old2(r: pd.DataFrame):
    r_mu = opt.matrix(r.mean(0).to_numpy())
    r_s = opt.matrix(r.cov().to_numpy())
    n = len(r_mu)
    G = -opt.matrix(np.eye(n))  # negative nxn identity matrix
    h = opt.matrix(0.0, (n, 1))  # n x 1 zero vector
    A = opt.matrix(1.0, (1, n))  # 1 x n 1-vector
    b = opt.matrix(1.0)  # [1.0]
    w = opt.solvers.qp(r_s, -r_mu, G, h, A, b)['x']
    w = w / sum(w)  # Allow for imperfections in fit, not in total allocation
    return pd.Series(w, index=r.columns)


def markowitz_port(r: pd.DataFrame):
    r_mu = opt.matrix(r.mean(0).to_numpy())
    r_mu = r_mu - np.mean(r_mu)
    r_s = opt.matrix(r.cov().to_numpy())
    n = len(r_mu)
    if n == 1:
        return pd.Series([1.0], index=r.columns)
    q = opt.matrix(0.0, (n, 1))  # n x 1 zero vector
    G = -opt.matrix(np.eye(n))  # negative nxn identity matrix
    h = opt.matrix(0.0, (n, 1))  # n x 1 zero vector
    A = opt.matrix(r_mu, (1, n))  # 1 x n 1-vector
    b = opt.matrix(1.0)  # [1.0]
    try:
        w = opt.solvers.qp(r_s, q, G, h, A, b)['x']
    except ValueError:
        w = np.ones(n)
    w = w / sum(w)  # Normalization
    return pd.Series(w, index=r.columns)


from pypfopt import EfficientFrontier


def markowitz_port_new(r: pd.DataFrame):
    r_mu = r.mean(0).to_numpy()
    r_s = r.cov().to_numpy()
    try:
        ef = EfficientFrontier(r_mu, r_s, verbose=True)
        ef.max_sharpe(risk_free_rate=0)
        w = ef.clean_weights()
        w = np.array(list(w.values()))
    except:
        try:
            ef = EfficientFrontier(r_mu, r_s, verbose=True)
            ef.min_volatility(0.1)
            w = ef.clean_weights()
            w = np.array(list(w.values()))
        except:
            w = np.ones(len(r_mu))
    w = w / sum(w)  # Normalization
    return pd.Series(w, index=r.columns)


df = pd.read_pickle('crsp_small.pickle')
# rets = df['RET'].unstack()

import glob

clusterings = ['base', 'km', 'kmm', 'agg', 'agg_mirr', 'dbscan', 'gm', 'ddm',
               'dpm', 'rand', 'naics']
full_names = ['No Clustering',
              'K-Means',
              'K-Medians',
              'Agglomerative (E)',
              'Agglomerative (mH)',
              'DBSCAN',
              'Gaussian Mixture',
              'Dirichlet Distribution',
              'Dirichlet Process',
              'Random',
              'NAICS']
short_names = ['No Clust', 'K-Mean', 'K-Med', 'Agg(E)', 'Agg(mH)',
               'DBSCAN', 'GM', 'DD', 'DP', 'Rand', 'NAICS']

# Performance of mean-reversion strat with forward-looking groups
last_rets = pd.Series(dtype=np.float64)
last_alpha_v = {x: pd.Series(dtype=np.float64) for x in clusterings}
performance = {x: pd.Series(dtype=np.float64) for x in clusterings}
for year, df_i in df.groupby(df.index.get_level_values(0).year):
    groups_i = pd.read_pickle(f'output_groups/{year}.pkl')
    rets_i = df_i.loc[
        df_i.index.isin(groups_i.index, level=1), "RET"].unstack()
    alpha = rets_i.shift()  # strategy is -logret/std of previous day
    # Only save the previous returns of invested assets
    last_rets = last_rets.loc[last_rets.index.intersection(alpha.columns)]
    alpha.iloc[0, :].loc[last_rets.index] = last_rets
    last_rets = rets_i.iloc[-1, :]  # Update the definition of last
    alpha = -np.log(1 + alpha)
    alpha = alpha / alpha.std()  # divide the std part
    alpha_pivot = alpha.copy(deep=True)  # Copy alpha to reuse for diff clusts

    # 'No group' neutralization
    alpha = alpha_pivot.add(-alpha_pivot.mean(axis=1), axis=0)
    alpha = alpha.div(0.5 * alpha.abs().sum(axis=1), axis=0)
    last_alpha_temp = alpha.iloc[-1, :]
    alpha = alpha.shift()
    last_alpha = last_alpha_v['base']
    last_alpha = last_alpha.loc[last_alpha.index.intersection(alpha.columns)]
    alpha.iloc[0, :].loc[last_alpha.index] = last_alpha
    last_alpha_v['base'] = last_alpha_temp
    performance['base'] = pd.concat([performance['base'],
                                     (alpha * rets_i).sum(1)])
    for x in clusterings[1:]:
        # Iterate the neutralization for each clustering
        alpha = alpha_pivot.copy(deep=True)
        alpha = alpha.add(
            -alpha.groupby(groups_i[x], axis=1).transform(np.mean))
        alpha = alpha.add(-alpha.mean(axis=1), axis=0)
        alpha = alpha.div(0.5 * alpha.abs().sum(axis=1), axis=0)
        last_alpha_temp = alpha.iloc[-1, :]
        alpha = alpha.shift()
        last_alpha = last_alpha_v[x]
        last_alpha = last_alpha.loc[
            last_alpha.index.intersection(alpha.columns)]
        alpha.iloc[0, :].loc[last_alpha.index] = last_alpha
        last_alpha_v[x] = last_alpha_temp
        performance[x] = pd.concat([performance[x],
                                    (alpha * rets_i).sum(1)])

# Performance of mean-reversion strat with a 1y delay in group assignations
last_rets = pd.Series(dtype=np.float64)
last_alpha_v = {x: pd.Series(dtype=np.float64) for x in clusterings}
performance2 = {x: pd.Series(dtype=np.float64) for x in clusterings}
for year, df_i in df.groupby(df.index.get_level_values(0).year):
    if year == 2010:
        continue
    groups_i = pd.read_pickle(f'output_groups/{year - 1}.pkl')
    rets_i = df_i.loc[
        df_i.index.isin(groups_i.index, level=1), "RET"].unstack()
    alpha = rets_i.shift()
    last_rets = last_rets.loc[last_rets.index.intersection(alpha.columns)]
    alpha.iloc[0, :].loc[last_rets.index] = last_rets
    last_rets = rets_i.iloc[-1, :]
    alpha = -np.log(1 + alpha)
    alpha = alpha / alpha.std()
    alpha_pivot = alpha.copy(deep=True)
    alpha = alpha.add(-alpha.mean(axis=1), axis=0)
    alpha = alpha.div(0.5 * alpha.abs().sum(axis=1), axis=0)
    last_alpha_temp = alpha.iloc[-1, :]
    alpha = alpha.shift()
    last_alpha = last_alpha_v['base']
    last_alpha = last_alpha.loc[last_alpha.index.intersection(alpha.columns)]
    alpha.iloc[0, :].loc[last_alpha.index] = last_alpha
    last_alpha_v['base'] = last_alpha_temp
    # rets_i = rets_i.replace(np.nan, -1)
    performance2['base'] = pd.concat([performance2['base'],
                                      (alpha * rets_i).sum(1)])
    for x in clusterings[1:]:
        alpha = alpha_pivot.copy(deep=True)
        alpha = alpha.add(
            -alpha.groupby(groups_i[x], axis=1).transform(np.mean))
        alpha = alpha.add(-alpha.mean(axis=1), axis=0)
        alpha = alpha.div(0.5 * alpha.abs().sum(axis=1), axis=0)
        last_alpha_temp = alpha.iloc[-1, :]
        alpha = alpha.shift()
        last_alpha = last_alpha_v[x]
        last_alpha = last_alpha.loc[
            last_alpha.index.intersection(alpha.columns)]
        alpha.iloc[0, :].loc[last_alpha.index] = last_alpha
        last_alpha_v[x] = last_alpha_temp
        performance2[x] = pd.concat([performance2[x],
                                     (alpha * rets_i).sum(1)])

# Performance of momentum strat lagged 1y and forward looking ##########

performance3 = {x: pd.Series(dtype=np.float64) for x in clusterings}
performance4 = {x: pd.Series(dtype=np.float64) for x in clusterings}
prev_w = {}
time_g = {x: [] for x in clusterings}
for year, df_i in df.groupby(df.index.get_level_values(0).year):
    print(year)
    groups_i = pd.read_pickle(f'output_groups/{year}.pkl')
    rets_i = df_i.loc[
        df_i.index.isin(groups_i.index, level=1), "RET"].unstack()
    t_start = time.time()
    w = markowitz_port_new(np.log(rets_i + 1))
    time_g['base'].append(time.time() - t_start)
    performance3['base'] = pd.concat([performance3['base'],
                                      (w * rets_i).sum(1)])
    if year > 2010:
        performance4['base'] = pd.concat([performance4['base'],
                                          (prev_w['base'] * rets_i).sum(1)])
    prev_w['base'] = w
    for x in clusterings[1:]:
        w_temp = pd.Series(index=rets_i.columns, dtype=np.float64)
        t_start = time.time()
        for _, df_ii in rets_i.groupby(groups_i[x], axis=1):
            w_i = markowitz_port_new(np.log(df_ii + 1))
            w_temp[w_i.index] = w_i
        rets_port = (w_temp * rets_i).groupby(groups_i[x], axis=1).sum()
        w_port = markowitz_port_new(np.log(rets_port + 1))
        time_g[x].append(time.time() - t_start)
        w = pd.Series(index=rets_i.columns, dtype=np.float64)
        for gi, w_i in w_temp.groupby(groups_i[x]):
            w[w_i.index] = w_i * w_port[gi]
        w = w / sum(w)
        performance3[x] = pd.concat([performance3[x],
                                     (w * rets_i).sum(1)])
        if year > 2010:
            performance4[x] = pd.concat([performance4[x],
                                         (prev_w[x] * rets_i).sum(1)])
        prev_w[x] = w

with open('performance.pkl', 'bw') as f:
    pickle.dump({'meanrev_fwd': performance,
                 'meanrev_lag': performance2,
                 'momtum_fwd': performance3,
                 'momtum_lag': performance4}, f)
with open('time_momentum.pkl', 'bw') as f:
    pickle.dump(time_g, f)

# PLOTS ######
with open('performance.pkl', 'br') as f:
    perf = pickle.load(f)
    performance = perf['meanrev_fwd']
    performance2 = perf['meanrev_lag']
    performance3 = perf['momtum_fwd']
    performance4 = perf['momtum_lag']
with open('time_momentum.pkl', 'br') as f:
    time_g = pickle.load(f)

# Cumulative Returns, Mean Reversion ######
perf_p = pd.concat(performance, axis=1)
perf_p = np.log(1 + perf_p)
perf_p.columns = full_names
perf_p = perf_p.iloc[:, np.argsort(perf_p.sum())[::-1]]
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(1, 2, figsize=[12.8, 5])
perf_p.cumsum().plot(ax=ax[0])
ax[0].set_ylabel("Cumulative Log-Returns")
ax[0].set_xlabel("Date")
ax[0].legend(fontsize=10, ncols=1, loc='upper left')
ax[0].set_title("Forward Looking Groups", fontsize=15)

perf_p = pd.concat(performance2, axis=1)
perf_p = np.log(1 + perf_p)
perf_p.columns = full_names
perf_p = perf_p.iloc[:, np.argsort(perf_p.sum())[::-1]]
perf_p.cumsum().plot(ax=ax[1])
# ax[1].set_ylabel("Cumulative Log-Returns")
ax[1].set_xlabel("Date")
ax[1].legend(fontsize=10, ncols=1, loc='upper left')
ax[1].set_title("Lagged Groups", fontsize=15)

fig.suptitle("Cumulative Log-Returns (2011-2021), Mean Reversion", fontsize=17)
fig.tight_layout()
plt.savefig("figs/mean_rev.pdf")
plt.show()

# Cumulative Returns, Momentum ######

perf_p = pd.concat(performance3, axis=1)
perf_p = np.log(1 + perf_p)
perf_p.columns = full_names
perf_p = perf_p.iloc[:, np.argsort(perf_p.sum())[::-1]]
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(1, 2, figsize=[12.8, 5])
perf_p.cumsum().plot(ax=ax[0])
ax[0].set_ylabel("Cumulative Log-Returns")
ax[0].set_xlabel("Date")
ax[0].legend(fontsize=10, ncols=1, loc='upper left')
ax[0].set_title("Forward Looking", fontsize=15)

perf_p = pd.concat(performance4, axis=1)
perf_p = np.log(1 + perf_p)
perf_p.columns = full_names
perf_p = perf_p.iloc[:, np.argsort(perf_p.sum())[::-1]]
perf_p.cumsum().plot(ax=ax[1])
# ax[1].set_ylabel("Cumulative Log-Returns")
ax[1].set_xlabel("Date")
ax[1].legend(fontsize=10, ncols=1, loc='upper left')
ax[1].set_title("Lagged", fontsize=15)

fig.suptitle("Cumulative Log-Returns (2011-2021), Momentum", fontsize=17)
fig.tight_layout()
plt.savefig("figs/momentum.pdf")
plt.show()

# Mean Returns and Sharpe, Mean Reversion ##############
perf_p = pd.DataFrame(
    [[performance[x].mean() * 252 * 100,
      performance[x].mean() / performance[x].std() * np.sqrt(252),
      ] for x in clusterings],
    index=full_names,
    columns=['Mean Yearly Returns (%)', 'Sharpe Ratio'])
perf_p = perf_p.sort_values('Sharpe Ratio', ascending=False)

plt.rcParams.update({'font.size': 15})

fig = plt.figure(layout='constrained', figsize=[8, 4.8])
subFigs = fig.subfigures(2, 1).flatten()
ax = subFigs[0].subplots(1, 2, sharey=True)
g = sns.barplot(x='Mean Yearly Returns (%)', y=perf_p.index, data=perf_p,
                ax=ax[0], palette="tab10", legend=False, hue=perf_p.index)
ax[0].tick_params(axis='y', labelsize=10)
ax[0].tick_params(axis='x', labelsize=10)
ax[0].set_xlabel(ax[0].get_xlabel(), fontsize=10)
g = sns.barplot(x='Sharpe Ratio', y=perf_p.index, data=perf_p,
                ax=ax[1], palette="tab10", legend=False, hue=perf_p.index)
ax[1].set_xlabel(ax[1].get_xlabel(), fontsize=10)
ax[0].set_ylabel('')
ax[1].tick_params(axis='x', labelsize=10)
subFigs[0].suptitle("                               "
                    "Forward Looking Groups", fontsize=13)
perf_p = pd.DataFrame(
    [[performance2[x].mean() * 252 * 100,
      performance2[x].mean() / performance2[x].std() * np.sqrt(252),
      ] for x in clusterings],
    index=full_names,
    columns=['Mean Yearly Returns (%)', 'Sharpe Ratio'])
perf_p = perf_p.sort_values('Sharpe Ratio', ascending=False)
ax = subFigs[1].subplots(1, 2, sharey=True)
g = sns.barplot(x='Mean Yearly Returns (%)', y=perf_p.index, data=perf_p,
                ax=ax[0], palette="tab10", legend=False, hue=perf_p.index)
ax[0].tick_params(axis='y', labelsize=10)
ax[0].tick_params(axis='x', labelsize=10)
ax[0].set_xlabel(ax[0].get_xlabel(), fontsize=10)
ax[0].set_ylabel('')

sns.barplot(x='Sharpe Ratio', y=perf_p.index, data=perf_p,
            ax=ax[1], palette="tab10", legend=False, hue=perf_p.index)
ax[1].tick_params(axis='x', labelsize=10)
ax[1].set_xlabel(ax[1].get_xlabel(), fontsize=10)
subFigs[1].suptitle("                                  "
                    "Lagged Groups", fontsize=13)
fig.suptitle("         "
             "Clustering Comparison (2011-2021), Mean Reversion", fontsize=14)
plt.savefig("figs/bars_sharpe_meanrev.pdf")
plt.show()

# Mean Returns and Sharpe, Momentum ##############

perf_p = pd.DataFrame(
    [[performance3[x].add(1).apply(np.log).mean() * 252 * 100,
      performance3[x].add(1).apply(np.log).mean() / performance3[x].add(
          1).apply(np.log).std() * np.sqrt(252),
      ] for x in clusterings],
    index=full_names,
    columns=['Mean Yearly Returns (%)', 'Sharpe Ratio'])
perf_p = perf_p.sort_values('Sharpe Ratio', ascending=False)

plt.rcParams.update({'font.size': 15})

fig = plt.figure(layout='constrained', figsize=[8, 4.8])
subFigs = fig.subfigures(2, 1).flatten()
ax = subFigs[0].subplots(1, 2, sharey=True)
g = sns.barplot(x='Mean Yearly Returns (%)', y=perf_p.index, data=perf_p,
                ax=ax[0], palette="tab10", legend=False, hue=perf_p.index)
ax[0].tick_params(axis='y', labelsize=10)
ax[0].tick_params(axis='x', labelsize=10)
ax[0].set_xlabel(ax[0].get_xlabel(), fontsize=10)
ax[0].set_ylabel('')
g = sns.barplot(x='Sharpe Ratio', y=perf_p.index, data=perf_p,
                ax=ax[1], palette="tab10", legend=False, hue=perf_p.index)
ax[1].set_xlabel(ax[1].get_xlabel(), fontsize=10)
ax[1].tick_params(axis='x', labelsize=10)
subFigs[0].suptitle("                               "
                    "Forward Looking", fontsize=13)
perf_p = pd.DataFrame(
    [[performance4[x].add(1).apply(np.log).mean() * 252 * 100,
      performance4[x].add(1).apply(np.log).mean() / performance4[x].add(
          1).apply(np.log).std() * np.sqrt(252),
      ] for x in clusterings],
    index=full_names,
    columns=['Mean Yearly Returns (%)', 'Sharpe Ratio'])
perf_p = perf_p.sort_values('Sharpe Ratio', ascending=False)
ax = subFigs[1].subplots(1, 2, sharey=True)
g = sns.barplot(x='Mean Yearly Returns (%)', y=perf_p.index, data=perf_p,
                ax=ax[0], palette="tab10", legend=False, hue=perf_p.index)
ax[0].tick_params(axis='y', labelsize=10)
ax[0].tick_params(axis='x', labelsize=10)
ax[0].set_xlabel(ax[0].get_xlabel(), fontsize=10)
ax[0].set_ylabel('')

sns.barplot(x='Sharpe Ratio', y=perf_p.index, data=perf_p,
            ax=ax[1], palette="tab10", legend=False, hue=perf_p.index)
ax[1].tick_params(axis='x', labelsize=10)
ax[1].set_xlabel(ax[1].get_xlabel(), fontsize=10)
subFigs[1].suptitle("                                  "
                    "Lagged", fontsize=13)
fig.suptitle("         "
             "Clustering Comparison (2011-2021), Momentum", fontsize=14)
# plt.tight_layout()
plt.savefig("figs/bars_sharpe_momentum.pdf")
plt.show()

# Mutual Information Score ########

clusterings_nn = clusterings[1:]
info_score_base = [[] for x in clusterings_nn]
for year in np.unique(df.index.get_level_values(0).year):
    groups_i = pd.read_pickle(f'output_groups/{year}.pkl')
    for i, c in enumerate(clusterings_nn):
        info_score_base[i].append(mutual_info_score(groups_i["naics"],
                                                    groups_i[c]))
info_score = [[np.mean(x), y] for x, y in zip(info_score_base, full_names[1:])]
perf_p = pd.DataFrame(info_score, columns=['Average Mutual Information Score',
                                           'Clustering'])
perf_p = perf_p.sort_values("Average Mutual Information Score")

plt.figure(figsize=[8.4, 2.4])
plt.rcParams.update({'font.size': 12})
# g1 = sns.barplot(x='Average Mutual Information Score', y="Clustering",
#                  data=perf_p, palette="tab10")
# g1.set(ylabel=None)
perf_p.set_index('Clustering').plot.barh(ax=plt.gca())
plt.gca().get_legend().remove()
plt.ylabel('')
plt.title("Average Mutual Information Score (2010-2021)")
plt.tight_layout()
plt.savefig("figs/MIS.pdf")
plt.show()

# Silhouette ########

clusterings_nn = clusterings[1:]
silh_score_base = [[] for x in clusterings_nn]
for year, df_i in df.groupby(df.index.get_level_values(0).year):
    groups_i = pd.read_pickle(f'output_groups/{year}.pkl')
    rets_i = df_i.loc[
        df_i.index.isin(groups_i.index, level=1), "RET"].unstack()
    rets_i = rets_i.div(rets_i.std(0), axis=1)
    for i, c in enumerate(clusterings_nn):
        silh_score_base[i].append(silhouette_score(rets_i.T,
                                                   groups_i[c]))

silh_score = [[np.mean(x), y] for x, y in zip(silh_score_base, full_names[1:])]
perf_p = pd.DataFrame(silh_score, columns=['Average Silhouette Score',
                                           'Clustering'])
perf_p = perf_p.sort_values("Average Silhouette Score")
plt.figure(figsize=[8.4, 2.4])
plt.rcParams.update({'font.size': 12})
# g1 = sns.barplot(x='Average Silhouette Score', y="Clustering",
#                  data=perf_p, palette="tab10")
# g1.set(ylabel=None)
perf_p.set_index('Clustering').plot.barh(ax=plt.gca())
plt.gca().get_legend().remove()
plt.ylabel('')
plt.title("Average silhouette distance (2010-2021)")
plt.tight_layout()
plt.savefig("figs/silh.pdf")
plt.show()

##### Time to fit Markowitz ###########
import pandas as pd
import matplotlib.pyplot as plt

ptimes = pd.DataFrame(time_g)
ptimes.columns = full_names
plt.figure(figsize=[8.4, 3.4])
ptimes.sum(0).sort_values(ascending=False).plot.bar(log=True)
plt.xticks(rotation=30, ha='right')
plt.ylabel("Seconds")
plt.ylim((100, 15000))
# plt.xlabel("Method")
plt.title("Time to Fit Markowitz Portfolio")
plt.tight_layout()
plt.savefig("figs/time_momentum.pdf")
plt.show()



import collections

corr_g = {x: collections.defaultdict(lambda: 0) for x in clusterings}
corr_lag_g = {x: collections.defaultdict(lambda: 0) for x in clusterings}
group_prev = None
rets_i_prev_g = None
corr_roundup = 1e2
for year, df_i in df.groupby(df.index.get_level_values(0).year):
    print(year)
    groups_i = pd.read_pickle(f'output_groups/{year}.pkl')
    rets_i = df_i.loc[
        df_i.index.isin(groups_i.index, level=1), "RET"].unstack()
    temp_corr = rets_i.corr()
    np.fill_diagonal(temp_corr.to_numpy(), np.nan)
    temp_corr = temp_corr.to_numpy().flatten()
    temp_corr = temp_corr[~np.isnan(temp_corr)]
    for y in temp_corr * corr_roundup:
        corr_g["base"][round(y)] += 1
    if year > 2010:
        rets_i_prev_g = df_i.loc[
            df_i.index.isin(group_prev.index, level=1), "RET"].unstack()
        temp_corr = rets_i_prev_g.corr()
        np.fill_diagonal(temp_corr.to_numpy(), np.nan)
        temp_corr = temp_corr.to_numpy().flatten()
        temp_corr = temp_corr[~np.isnan(temp_corr)]
        for y in temp_corr * corr_roundup:
            corr_lag_g["base"][round(y)] += 1
    for x in clusterings[1:]:
        for _, df_ii in rets_i.groupby(groups_i[x], axis=1):
            temp_corr = df_ii.corr()
            np.fill_diagonal(temp_corr.to_numpy(), np.nan)
            temp_corr = temp_corr.to_numpy().flatten()
            temp_corr = temp_corr[~np.isnan(temp_corr)]
            for y in temp_corr * corr_roundup:
                corr_g[x][round(y)] += 1
        if year > 2010:
            for _, df_ii in rets_i_prev_g.groupby(group_prev[x], axis=1):
                temp_corr = df_ii.corr()
                np.fill_diagonal(temp_corr.to_numpy(), np.nan)
                temp_corr = temp_corr.to_numpy().flatten()
                temp_corr = temp_corr[~np.isnan(temp_corr)]
                for y in temp_corr * corr_roundup:
                    corr_lag_g[x][round(y)] += 1
    group_prev = groups_i

corr_lag_g2 = {k: dict(v) for k, v in corr_lag_g.items()}
corr_g2 = {k: dict(v) for k, v in corr_g.items()}
to_save = [corr_g2, corr_lag_g2]
with open("corr_g_lag_g.pkl", "wb") as f:
    pickle.dump(to_save, f)


with open("corr_g_lag_g.pkl", "rb") as f:
    (corr_g2, corr_lag_g2) = pickle.load(f)

import gc
corr_df = pd.DataFrame()
for i, x in enumerate(clusterings):
    corr_df_temp = pd.DataFrame(
        {"Correlation": [xx / corr_roundup for xx in corr_g2[x].keys()],
         "y": corr_g2[x].values()})
    corr_df_temp.y = np.round(corr_df_temp.y / 100)
    repeat_df = corr_df_temp['Correlation'].repeat(corr_df_temp['y'])
    repeat_df.loc[len(repeat_df)] = -1
    repeat_df.loc[len(repeat_df)] = 1
    repeat_df = pd.DataFrame(repeat_df)
    repeat_df["Cluster"] = short_names[i]
    corr_df = pd.concat([corr_df, repeat_df])
    gc.collect()

import seaborn as sns

sns.violinplot(corr_df, x="Correlation", y="Cluster", palette="tab10",
               legend=False, hue="Cluster")
plt.title("Intra-Group Correlation Distribution")
plt.ylabel('')
plt.tight_layout()
plt.savefig("figs/ig_corr.pdf")
plt.show()

corr_df = pd.DataFrame()
for i, x in enumerate(clusterings):
    corr_df_temp = pd.DataFrame(
        {"Correlation": [xx / corr_roundup for xx in corr_lag_g2[x].keys()],
         "y": corr_lag_g2[x].values()})
    corr_df_temp.y = np.round(corr_df_temp.y / 100)
    repeat_df = corr_df_temp['Correlation'].repeat(corr_df_temp['y'])
    repeat_df.loc[len(repeat_df)] = -1
    repeat_df.loc[len(repeat_df)] = 1
    repeat_df = pd.DataFrame(repeat_df)
    repeat_df["Cluster"] = short_names[i]
    corr_df = pd.concat([corr_df, repeat_df])

sns.violinplot(corr_df, x="Correlation", y="Cluster", palette="tab10",
               legend=False, hue="Cluster")
plt.title("Intra-Group Correlation Distribution, Lagged")
plt.ylabel('')
plt.tight_layout()
plt.savefig("figs/ig_corr_lag.pdf")
plt.show()
