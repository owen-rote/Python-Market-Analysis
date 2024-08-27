# Author: owen-rote
# Purpose: Contains functions to download S&P 500 tickers, fetch historical trading data
#          from Yahoo Finance, compile the data into a single dataframe, and generate a
#          correlation heatmap of stock prices or returns.

# Functions shall be called in the order fetch_yahoo_data() -> compile_data() -> visualize_data()

import bs4 as bs
import datetime as dt
import json
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import yfinance as yf

style.use("ggplot")


def save_sp500_tickers() -> list:
    """Creates a JSON file with a list of all S&P 500 tickers

    This function scrapes the Wikipedia page for the S&P 500 index to get the list of
    current tickers, saves them to a JSON file, and returns the list of tickers.

    Helper function for fetch_yahoo_data()

    Returns:
        list: A list of all S&P 500 tickers
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
    """Downloads historical stock data for S&P 500 tickers from Yahoo Finance.

    This function iterates through the list of S&P 500 tickers and downloads historical
    stock data, saving each ticker's data to a CSV file in the 'stock_dfs' directory.

    Args:
        reload_sp500 (bool, optional): If True, re-downloads the updated list of S&P 500 tickers. Defaults to False.
        reload_data (bool, optional): If True, re-downloads all stock data. Defaults to False.
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

    # Reload data if desired
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


def compile_data() -> None:
    """Compiles adjusted closing prices of S&P 500 stocks into a single dataframe.

    This function reads the historical data of each S&P 500 stock from CSV files, extracts the
    adjusted closing prices, and compiles them into one dataframe. The resulting dataframe is
    saved to 'sp500_joined_closes.csv'.
    """
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

    print("Compilation complete")
    main_df.to_csv("sp500_joined_closes.csv")


def visualize_data(pct_change=False) -> None:
    """Generates a heatmap correlation table of S&P 500 stock prices OR returns

    Must have available pre-compiled dataset from compile_data()

    Args:
        pct_change (bool, optional): If True, generates based on returns rather than price.
            Returns tend to follow normal distrubution and prices don't. Defaults to False.
    """
    df = pd.read_csv("sp500_joined_closes.csv", parse_dates=["Date"], index_col="Date")

    # Generate correlation table
    if pct_change:
        df_corr = df.pct_change().corr()
    else:
        df_corr = df.corr()

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # =============== Generate heatmap on a grid ===============
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)

    # Add tickmarks
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # Add labels
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)

    # Set range from -1 to 1
    heatmap.set_clim(-1, 1)

    plt.tight_layout
    plt.show()


if __name__ == "__main__":
    print("Initating S&P correlation table generator.\n")
    # Ask if user wants most recent stock data
    user_input = ''
    while user_input.lower() != 'y' and user_input.lower() != 'n':
        user_input = input("Would you like to fetch the most recent stock data AND re-compile data? \nThis will take a few minutes. (y/n): ")

        if user_input == "y":
            fetch_yahoo_data(True, True)
            compile_data()

    user_input = ''
    while user_input.lower() != 'y' and user_input.lower() != 'n':
        user_input = input("Would you like to visualize percent change correlation rather than price? (y/n): ")

        if user_input == 'y':
            visualize_data(True)
        elif user_input == 'n':
            visualize_data(False)
