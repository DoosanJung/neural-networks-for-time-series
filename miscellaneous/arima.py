#!/usr/bin/env python
'''
'''
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def get_ts(show=None, **kwargs):
    ts= pd.read_json('data/logins.json', typ='series')
    ts = pd.Series(1, ts) # add constant

    day = kwargs.pop("Day", False)
    hour = kwargs.pop("Hour", False)
    if kwargs:
        raise TypeError("Unexpected **kwargs: {}".format(kwargs))

    if day == True:
        ts = ts.resample(rule='D', how='count')
        if show == True:
            ts.plot(figsize=(12,8), title='daily number of logins')
            plt.show()
    elif hour == True:
        ts = ts.resample(rule='H', how='count')
        if show == True:
            ts.plot(figsize=(12,8), title='hourly number of logins')
            plt.show()
    return ts

def EDA(ts, no_diff=None, first_diff=None, first_season_diff=None, ols_residual=None):
    # highlight this strong seasonal component
    df_day = get_df(ts)
    df_day['count'].plot(figsize=(12,8),title='highlighting this storing seasonal component')
    plt.fill_between(df_day.index, df_day['count'], where=df_day['weekend'])
    plt.show()

    # Plot ACF/PACF
    if no_diff == True:
        plot_acf_pacf(ts_day, lags=28)
        plt.show()
    elif first_diff == True:
        # Plot ACF/PACF after detrending
        # 1st difference can remove linear trend
        ts_1st_diff = detrend_ts(ts_day, first_diff = True)
        plot_acf_pacf(ts_1st_diff, lags=28)
        plt.show()
    elif first_season_diff == True:
        # 1st seasonal difference can remove linear seasonal trend
        ts_1st_seasonal_diff = detrend_ts(ts_day, first_seasonal_diff = True)
        plot_acf_pacf(ts_1st_seasonal_diff, lags=28)
        plt.show()
    elif ols_residual == True:
        # treat the residuals as your new time series
        ts_ols_residual = detrend_ts(ts_day, ols_residual = True)
        plot_acf_pacf(ts_ols_residual, lags=28)
        plt.show()
    else:
        pass

def get_df(ts):
    df = pd.DataFrame(ts).rename(columns={0: 'count'})
    df['dayofweek'] = pd.DatetimeIndex(df.index).weekday
    df['weekend'] = df['dayofweek'] >= 5
    return df

def plot_acf_pacf(data, lags):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(data.dropna(), lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(data.dropna(), lags=lags, ax=ax2)

def detrend_ts(ts, **kwargs):
    first_diff = kwargs.pop("first_diff", False)
    first_seasonal_diff = kwargs.pop("first_seasonal_diff", False)
    ols_residual = kwargs.pop("ols_residual", False)
    if kwargs:
        raise TypeError("Unexpected **kwargs: {}".format(kwargs))

    if first_diff == True:
        ts_detrend = ts.diff(periods=1)

    elif first_seasonal_diff == True:
        ts_detrend = ts.diff(periods=7)

    elif ols_residual == True:
        y = ts.values
        X = range(1,ts.shape[0]+1)
        model = sm.OLS(y, sm.add_constant(X)).fit()
        df_day = get_df(ts_day)
        ts_detrend= pd.Series(model.resid, index=df_day.index)

    return ts_detrend


if __name__=="__main__":
    ts = get_ts()
    ts_hour = get_ts(Hour=True)
    ts_day = get_ts(show=True, Day=True)
    print("There is a strong seasonal component in the data")

    # EDA
    EDA(ts_day, no_diff = None, first_diff=None, first_season_diff=True, ols_residual=None)

    # Seasoal ARIMA model
    # The order argument is a tuple of the form (AR specification, Integration order, MA specification)
    # The seasonal order argument is a tuple of the form 
    # (Seasonal AR specification, Seasonal Integration order, Seasonal MA, Seasonal periodicity).
    model=sm.tsa.SARIMAX(ts_day, order=(1,1,0), seasonal_order=(0,1,0,7)).fit()
    model.summary()
    plot_acf_pacf(model.resid, lags=28)
    plt.show()
