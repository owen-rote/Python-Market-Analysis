import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import yfinance as yf

style.use("ggplot")

start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()
df = yf.download('BTC-USD', start, end)

print(df.head())

