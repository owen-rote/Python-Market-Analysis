# Author: owen-rote
# Purpose: Contains functions for preprocessing stock data and training
#          a buy/sell/hold prediction model using a voting classifier.

# Note: Run 'sp500 data compiler.py' and select YES for downloading recent data.
# in order to download required datasets.

from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def process_data_for_labels(ticker):
    """Calculate the percentage change of stock prices over future days.

    This function reads stock data, calculates the percentage change in the stock price
    over a specified number of future days, and returns a list of tickers and the updated dataframe.

    Args:
        ticker (str): The stock ticker symbol

    Returns:
        list: List of processed tickers
        np.DataFrame: New dataframe with calculated percentage changes
    """
    future_days = 7  # How many days do we have to make x%
    df = pd.read_csv("sp500_joined_closes.csv", index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, future_days + 1):
        df["{}_{}d".format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[
            ticker
        ]

    df.fillna(0, inplace=True)

    return tickers, df


def buy_sell_hold(*args):
    """Label each price change as buy, sell, or hold.

    This function labels each input percentage change as buy, sell, or hold based on a
    threshold. If the percentage change is greater/less/within  the threshold, the function
    returns a 1/-1/0 or buy/sell/hold respectively.


    Returns:
        int: 1 for buy, -1 for sell, 0 for hold.
    """
    # Signal a buy or sell
    cols = [c for c in args]
    requirement = 0.02  # Threshold: if stock changes by 2%
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1

    return 0


def extract_featuresets(ticker):
    """Extracts nominal features and labels for stock prediction

    This function processes the stock data to extract features and labels for the
    buy/sell/hold prediction model. It returns the feature set, labels, and the updated dataframe.

    Args:
        ticker (str): The stock ticker symbol

    Returns:
        np.ndarray: Array of feature values.
        np.ndarray: Array of target labels.
        pd.DataFrame: Updated dataframe with features and labels.
    """
    tickers, df = process_data_for_labels(ticker)

    # Map a 1/-1/0 label from buy_sell_hold()
    df["{}_target".format(ticker)] = list(
        map(
            buy_sell_hold,
            df["{}_1d".format(ticker)],
            df["{}_2d".format(ticker)],
            df["{}_3d".format(ticker)],
            df["{}_4d".format(ticker)],
            df["{}_5d".format(ticker)],
            df["{}_6d".format(ticker)],
            df["{}_7d".format(ticker)],
        )
    )

    vals = df["{}_target".format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print("Data spread: ", Counter(str_vals))
    # Clean improper values
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    # Add percent change
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df["{}_target".format(ticker)].values

    return X, y, df


def do_ml(ticker):
    """Train and evaluate a voting classifier model for stock prediction.

    This function trains a voting classifier composed of LinearSVC, KNeighborsClassifier,
    and RandomForestClassifier, evaluates the model, and prints the accuracy and
    distribution of the predictions.

    Args:
        ticker (str): The stock ticker symbol to generate the prediction model for.

    Returns:
        float: The accuracy score of the model on the test set.
    """
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Add voting classifier
    clf = VotingClassifier(
        [
            ("lsvc", LinearSVC()),
            ("knm", KNeighborsClassifier()),
            ("rfor", RandomForestClassifier()),
        ]
    )

    # Display accuracy
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print("Accuracy:", confidence)

    # Count predictions for each label
    predicted_counts = Counter(clf.predict(X_test))
    # Convert counts to int to avoid printing 'np.int64(x)'
    predicted_counts_int = {int(k): v for k, v in predicted_counts.items()}
    print("Predicted spread:", predicted_counts_int)

    return confidence


if __name__ == "__main__":
    while True:
        try:
            choice = input("Please enter a S&P 500 ticker to predict (EX: 'TSLA'): ")
            do_ml(choice.upper())
            break
        except Exception:
            print("ERROR: Ticker invalid.")

