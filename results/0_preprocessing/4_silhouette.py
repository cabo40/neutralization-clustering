import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pickle

df = pd.read_pickle('../../crsp_small.pickle')
silh = {}
naics_diff = []
nclusters = range(2, 51)
for year, df_i in df.groupby(df.index.get_level_values(0).year):
    rets_i = df_i['RET'].unstack()
    naics_i = df_i['NAICS'].astype(str).str[:2]
    naics_i = naics_i[~naics_i.isin(['99', '67'])].unstack()
    rets_i = rets_i.dropna(axis=1)
    naics_i = naics_i.dropna(axis=1)
    permno_i = naics_i.columns[naics_i.nunique(axis=0, dropna=False) == 1]
    rets_i = rets_i[rets_i.columns.intersection(permno_i)]
    logrets_i = np.log(1 + rets_i)
    nlogrets_i = logrets_i / logrets_i.std()
    nnlogrets_i = nlogrets_i.add(-nlogrets_i.mean(1), axis=0).div(
        nlogrets_i.std(1), axis=0)
    permno_i = rets_i.columns
    naics_i = naics_i[permno_i].iloc[0, :]
    sill = []
    for _ in range(20):
        sil = []
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        for k in nclusters:
            km = KMeans(k, n_init=1)
            kn = km.fit_predict(nnlogrets_i.T)
            sil.append(silhouette_score(nnlogrets_i.T, kn))
        sill.append(sil)
    silh[year] = sill
    with open("../../silh.pkl", "wb") as f:
        pickle.dump(silh, f)

with open("../../silh.pkl", "rb") as f:
    silh = pickle.load(f)

plt.figure(figsize=[6.4, 4.4])
plt.rcParams.update({'font.size': 15})
for key in silh.keys():
    plt.plot(nclusters, np.array(silh[key]).max(axis=0), label=key,
             alpha=(key - 2010 + 5) / 15)
plt.xlabel("Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score, Several Years")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig("figs/silhouette-kmeans-normalized.pdf")
plt.show()

plt.figure(figsize=[6.4, 4.4])
plt.plot(nclusters,
         np.array(list(silh.values())).max(axis=0).mean(axis=0))
plt.xlabel("Clusters")
plt.ylabel("Silhouette Score")
plt.title("Average Silhouette Score")
plt.tight_layout()
plt.savefig("figs/silhouette-kmeans-normalized-year-average.pdf")
plt.show()

# plt.figure(figsize=[4.8, 3.6])
plt.rcParams.update({'font.size': 15})
ax = nlogrets_i.plot(legend=False, xlabel="Date",
                     title="U.S. Stock Market")
ax.xaxis.set_ticks([x.get_loc() for x in ax.xaxis.get_major_ticks()[:-1]] +
                   [ax.xaxis.get_major_ticks()[-1].get_loc() - 1])
ax.set_ylabel("Scaled Log-Returns")
plt.tight_layout()
plt.savefig('figs/logret.png')
plt.show()

plt.rcParams.update({'font.size': 15})
sm.qqplot(nlogrets_i.unstack(),
          fit=True, line='45')
plt.title("Q-Q Plot of Scaled Log-Returns")
plt.tight_layout()
plt.savefig('figs/qqplotstd.png')
plt.show()
