# Author: owen-rote
# Purpose: Demonstrates different charting options with yfinance stock data.
#          Downloads and charts live stock data from desired ticker

import datetime as dt
import os
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib import style
import pandas as pd
import yfinance as yf

style.use("ggplot")

# Download spy data
# =================
if not os.path.exists('spy.csv'):
    start = dt.datetime(2020, 1, 1)
    end = dt.datetime.now()
    df = yf.download('SPY', start, end)
    df.to_csv('spy.csv')

# Initialize dataframe
df = pd.read_csv("spy.csv", parse_dates=True, index_col=0)

# Create 100 moving average col (Avg of last 100 days' price)
df["100ma"] = df["Adj Close"].rolling(window=100, min_periods=0).mean()
# min_periods adjusts for first 100 rows

# Price x 100ma x Volume plot
# ===========================
# ax1.plot(df.index, df["Adj Close"])
# ax1.plot(df.index, df["100ma"])
# ax2.bar(df.index, df["Volume"])

# Create resampled dataframe (Open high low close)
df_ohlc = df["Adj Close"].resample("10D").ohlc()
df_volume = df["Volume"].resample("10D").sum()
df_ohlc["volume"] = df_volume

mpf.plot(
    df_ohlc,
    type="candle",
    title="",
    ylabel="",
    ylabel_lower="",
    figratio=(25, 10),
    figscale=1,
    mav=50,
    volume=True,
)

plt.show()
