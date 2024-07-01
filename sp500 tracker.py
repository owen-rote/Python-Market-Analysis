# Author: owen-rote
# Purpose:

import bs4 as bs
import datetime as dt
import os
import pandas as pd
import yfinance as yf
import json
import requests


def save_sp500_tickers() -> list:
    """Creates json file with a list of all s&p 500 tickers

    Returns:
        list: All s&p 500 tickers
    """
    response = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = bs.BeautifulSoup(response.text, "lxml")
    table = soup.find("table", {"class": "wikitable sortable"})
    tickers = []
    for row in table.findAll("tr")[1:]:
        ticker = row.find("td").text.strip()
        tickers.append(ticker)

    with open("sp500tickers.txt", "w") as file:
        json.dump(tickers, file)

    return tickers


def fetch_yahoo_data(reload_sp500=False, reload_data=False) -> None:
    """Iterates through tickers and saves a csv of the data in stock_dfs/

    Args:
        reload_sp500 (bool, optional): Re-downloads updated s&p 500 tickers. Defaults to False.
        reload_data (bool, optional): Re-downloads all stock data. Defaults to False.
    """

    # Reload tickers if required
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.txt", "r") as f:
            tickers = json.load(f)

    # Create dataframes directory
    if not os.path.exists("stock_dfs"):
        os.makedirs("stock_dfs")

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime.now()

    # Reload data if required
    if reload_data:
        for ticker in tickers:
            print("Reloading: {}".format(ticker))
            df = yf.download(ticker, start, end)
            df.to_csv("stock_dfs/{}.csv".format(ticker))

        return

    # Loop tickers and save data
    for ticker in tickers:
        if not os.path.exists("stock_dfs/{}.csv".format(ticker)):
            print("Loading {}".format(ticker))
            df = yf.download(ticker, start, end)
            df.to_csv("stock_dfs/{}.csv".format(ticker))
        else:
            print("Already have {}".format(ticker))


# fetch_yahoo_data()


def compile_data() -> None:
    with open("sp500tickers.txt", "r") as f:
        tickers = json.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv("stock_dfs/{}.csv".format(ticker))
        df.set_index("Date", inplace=True)
        df.rename(columns={"Adj Close": ticker}, inplace=True)
        df.drop(["Open", "High", "Low", "Close", "Volume"], axis=1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how="outer")

        if count % 10 == 0:
            print(count)

    print(main_df.head())

    main_df.to_csv("sp500_joined_closes.csv")


compile_data()

