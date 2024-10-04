import pandas as pd
df = pd.read_csv('crsp_small.csv', parse_dates=[1])
df = df.set_index(["date", "PERMNO"])
df = df.sort_index()

df.to_pickle('crsp_small.pickle')
