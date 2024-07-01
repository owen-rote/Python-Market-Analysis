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
df["100ma"] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
# min_periods adjusts for first 100 rows

print(df.head())
