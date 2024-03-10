import pandas as pd
import matplotlib.pyplot as plt

def rolling_mean(series, window, plot=0):
    """
    Calculate the rolling mean of a given series and optionally plot the original series and its rolling mean.

    Parameters:
    - series (pd.Series): The series to calculate the rolling mean of.
    - window (int): The window size for the rolling mean calculation.
    - plot (int, optional): If positive, plot the original series and its rolling mean. Defaults to 0 (no plot).

    Returns:
    - pd.Series: The rolling mean of the series.
    """
    # Calculate rolling mean
    rolling_mean = series.rolling(window=window).mean()

    # Plot if plot argument is positive
    if plot > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(series, linestyle=':', label='Original Series')
        plt.plot(rolling_mean, label='Rolling Mean', linewidth=2)
        plt.title('Original Series and Rolling Mean')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

    return rolling_mean
