import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_pickle('crsp_small.pickle')
ret = df['RET'].loc['2020-01-01':].unstack()
grp = df['NAICS'].loc['2020-01-01':].unstack()
stable = grp.columns[grp.nunique(axis=0, dropna=False) == 1]
ret = ret.loc[:, stable]
grp = grp.loc[:, stable].iloc[0, :]

ax = np.log(1 + ret).plot(legend=False, xlabel="date",
                          title="U.S. Stock Market")
ax.xaxis.set_ticks([x.get_loc() for x in ax.xaxis.get_major_ticks()[:-1]] +
                   [ax.xaxis.get_major_ticks()[-1].get_loc() - 1])
ax.set_ylabel("Log-Returns")
plt.tight_layout()
plt.savefig('figs/logret.png')
plt.show()

temp = np.log(1 + ret)
temp = temp.add(-temp.mean(axis=0), axis=1)
temp = temp.div(temp.std(axis=0), axis=1)
temp = temp.unstack().reset_index(drop=True)
sm.qqplot(temp, fit=True, line='45')
plt.title("Q-Q plot")
plt.tight_layout()
plt.savefig('figs/qqplotstd.png')
plt.show()
