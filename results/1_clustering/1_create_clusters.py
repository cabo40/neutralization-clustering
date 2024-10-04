from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import DBSCAN, kmeans_plusplus
from scipy.spatial.distance import squareform, pdist
from pyclustering.cluster import kmedians
import pandas as pd
import numpy as np
import tqdm
import time


def kmedians_wrap(n_clusters, df):
    centers, _ = kmeans_plusplus(
        df.to_numpy(), n_clusters, random_state=0)
    kkmedians = kmedians.kmedians(df, centers)
    kkmedians.process()
    clusters = kkmedians.get_clusters()
    ret = np.empty(df.shape[0], dtype=np.int16)
    ret[:] = -1
    for group, elements in enumerate(clusters):
        ret[elements] = group
    return ret


# Create the groups
k = 22
v_times = []
df = pd.read_pickle('crsp_small.pickle')
for year, df_i in tqdm.tqdm(df.groupby(df.index.get_level_values(0).year)):
    times = {}
    rets_i = df_i['RET'].unstack()
    naics_i = df_i['NAICS'].astype(str).str[:2]
    naics_i = naics_i[~naics_i.isin(['99', '67'])].unstack()
    rets_i = rets_i.dropna(axis=1)
    naics_i = naics_i.dropna(axis=1)
    permno_i = naics_i.columns[naics_i.nunique(axis=0, dropna=False) == 1]
    rets_i = rets_i[rets_i.columns.intersection(permno_i)]
    permno_i = rets_i.columns
    naics_i = naics_i[permno_i].iloc[0, :]

    logrets_i = np.log(1 + rets_i)
    scaled_logrets_i = logrets_i / logrets_i.std()
    X_i = scaled_logrets_i

    start_time = time.time()
    km_i = KMeans(n_clusters=k, n_init=10).fit_predict(X_i.T)
    end_time = time.time()
    times['km'] = end_time - start_time

    start_time = time.time()
    kmed_i = kmedians_wrap(k, X_i.T)
    end_time = time.time()
    times['kmm'] = end_time - start_time

    start_time = time.time()
    agg_euc_i = AgglomerativeClustering(k, metric="l2",
                                        linkage="complete").fit_predict(X_i.T)
    end_time = time.time()
    times['agg'] = end_time - start_time

    start_time = time.time()
    m_hamming = squareform(
        pdist((X_i > 0).T,
              lambda u, v: min((u ^ v).sum(), (u ^ ~v).sum()) / len(u)))
    agg_m_euc_i = AgglomerativeClustering(
        n_clusters=k, metric='precomputed', linkage='complete'
    ).fit_predict(m_hamming)
    end_time = time.time()
    times['agg_mirr'] = end_time - start_time

    start_time = time.time()
    dbscan_i = DBSCAN(eps=252 * .05, min_samples=10).fit_predict(X_i.T)
    end_time = time.time()
    times['dbscan'] = end_time - start_time

    start_time = time.time()
    gm_i = GaussianMixture(n_components=k).fit_predict(X_i.T)
    end_time = time.time()
    times['gm'] = end_time - start_time

    start_time = time.time()
    ddm_i = BayesianGaussianMixture(
        n_components=k,
        weight_concentration_prior_type='dirichlet_distribution',
        weight_concentration_prior=1,
        mean_prior=np.zeros(X_i.shape[0]),
        mean_precision_prior=1,
        degrees_of_freedom_prior=X_i.shape[0],
        covariance_prior=np.identity(X_i.shape[0])
    ).fit_predict(X_i.T)
    end_time = time.time()
    times['ddm'] = end_time - start_time

    start_time = time.time()
    dpm_i = BayesianGaussianMixture(n_components=k,
                                    weight_concentration_prior=2.932,
                                    mean_prior=np.zeros(X_i.shape[0]),
                                    mean_precision_prior=1,
                                    degrees_of_freedom_prior=X_i.shape[0],
                                    covariance_prior=np.identity(X_i.shape[0])
                                    ).fit_predict(X_i.T)
    end_time = time.time()
    times['dpm'] = end_time - start_time

    start_time = time.time()
    rand_i = np.random.randint(k, size=X_i.shape[1])
    end_time = time.time()
    times['rand'] = end_time - start_time

    groups_i = pd.DataFrame(
        {
            'km': km_i,
            'kmm': kmed_i,
            'agg': agg_euc_i,
            'agg_mirr': agg_m_euc_i,
            'dbscan': dbscan_i,
            'gm': gm_i,
            'ddm': ddm_i,
            'dpm': dpm_i,
            'rand': rand_i,
            'naics': naics_i},
        index=X_i.columns
    )
    v_times.append(times)

import matplotlib.pyplot as plt
s = pd.Series({a: np.mean([v_times[x][a] for x in [0, 1]]) for a in
           v_times[0].keys()})
s = s.rename({'rand': 'Random',
          'dbscan': 'DBSCAN',
          'km': '$k$-means',
          'kmm': '$k$-medians',
          'agg': 'Agglomerative(E)',
          'agg_mirr': 'Agglomerative(mH)',
          'ddm': 'DDM',
          'gm': 'GMM',
          'dpm': 'DPM'})
s = s.sort_values()
ax = s.plot.barh(log=True, figsize=[8.4, 2.4])
plt.title('Mean running time to fit')
plt.tight_layout()
plt.savefig('figs/clus_times.pdf')
plt.show()