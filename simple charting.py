import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import yfinance as yf

style.use("ggplot")

# Uncomment to download csv
# =========================
# start = dt.datetime(2020, 1, 1)
# end = dt.datetime.now()
# df = yf.download('SPY', start, end)
# df.to_csv('spy.csv')

df = pd.read_csv("spy.csv", parse_dates=True, index_col=0)


# Data display options
# ====================
# print(df.tail())
# print(df[["Open", "High"]].head())
# print(df["Adj Close"].head())
# df["Adj Close"].plot()
# plt.show()

# Create 100 moving average col (Avg of last 100 days' price)
df["100ma"] = df["Adj Close"].rolling(window=100, min_periods=0).mean()
# min_periods adjusts for first 100 rows

print(df.head())

#                      (size) (start pt)
ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6, 1), (0, 0), rowspan=1, colspan=1, sharex=ax1)

ax1.plot(df.index, df["Adj Close"])
ax1.plot(df.index, df["100ma"])
ax2.bar(df.index, df["Volume"])

plt.show()
