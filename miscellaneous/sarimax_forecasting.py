#!/usr/bin/env python
'''
reference: http://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_stata.html
'''
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO

class Forecasting():
    def __init__(self, data):
        self.data = data

    def get_variables(self):
        # Variables
        self.endog = self.data.loc['1959':, 'consump']
        self.exog = sm.add_constant(self.data.loc['1959':, 'm2'])

    def first_stage(self):
        # Fit the model (using data from 1959 to 1978)
        model = sm.tsa.statespace.SARIMAX(self.endog.loc[:'1978-01-01'], \
                                        exog=self.exog.loc[:'1978-01-01'], \
                                        order=(1,0,1))
        self.fit_res = model.fit(disp=False)
        print(self.fit_res.summary())
        print("estimated parameters..")
        print(self.fit_res.params)

    def second_stage_one_step_ahead(self):
        model = sm.tsa.statespace.SARIMAX(self.endog, exog=self.exog, order=(1,0,1))
        res = model.filter(self.fit_res.params)
        self.predict, self.predict_ci = self.__one_step_ahead_forecasting(res)

    def second_stage_dynamic(self):
        model = sm.tsa.statespace.SARIMAX(self.endog, exog=self.exog, order=(1,0,1))
        res = model.filter(self.fit_res.params)
        self.predict_dy, self.predict_dy_ci = self.__dynamic_forecasting(res)

    def __one_step_ahead_forecasting(self, res):
        '''
        In-sample one-step-ahead predictions
        One-step-ahead prediction uses the true values of the endogenous values at each step
        to predict the next in-sample value
        '''
        predict = res.get_prediction()
        predict_ci = predict.conf_int()
        return predict, predict_ci

    def __dynamic_forecasting(self, res):
        '''
        Dynamic predictions starting in the first quarter of 1978.
        Dynamic predictions use one-step-ahead prediction up to some point in the dataset;
        after that, the previous predicted endogenous values are used
        in place of the true endogenous values for each new predicted element.
        '''
        predict_dy = res.get_prediction(dynamic='1978-01-01')
        predict_dy_ci = predict_dy.conf_int()
        return predict_dy, predict_dy_ci

    def plot_forecasting(self):
        fig, ax = plt.subplots(figsize=(9,4))
        ax.set(title='Personal consumption', xlabel='Date', ylabel='Billions of dollars')

        # Plot data points
        self.data.ix['1977-07-01':, 'consump'].plot(ax=ax, style='o', label='Observed')

        # Plot predictions
        self.predict.predicted_mean.ix['1977-07-01':].plot(ax=ax, style='r--', label='One-step-ahead forecast')
        ci = self.predict_ci.ix['1977-07-01':]
        ax.fill_between(ci.index, ci.ix[:,0], ci.ix[:,1], color='r', alpha=0.1)
        self.predict_dy.predicted_mean.ix['1977-07-01':].plot(ax=ax, style='g', label='Dynamic forecast (1978)')
        ci = self.predict_dy_ci.ix['1977-07-01':]
        ax.fill_between(ci.index, ci.ix[:,0], ci.ix[:,1], color='g', alpha=0.1)

        legend = ax.legend(loc='lower right')
        plt.show()

    def plot_forecasting_error(self):
        fig, ax = plt.subplots(figsize=(9,4))
        ax.set(title='Forecast error', xlabel='Date', ylabel='Forecast - Actual')

        # In-sample one-step-ahead predictions and 95% confidence intervals
        predict_error = self.predict.predicted_mean - self.endog
        predict_error.ix['1977-10-01':].plot(ax=ax, label='One-step-ahead forecast')
        ci = self.predict_ci.ix['1977-10-01':].copy()
        ci.iloc[:,0] -= self.endog.loc['1977-10-01':]
        ci.iloc[:,1] -= self.endog.loc['1977-10-01':]
        ax.fill_between(ci.index, ci.ix[:,0], ci.ix[:,1], alpha=0.1)

        # Dynamic predictions and 95% confidence intervals
        predict_dy_error = self.predict_dy.predicted_mean - self.endog
        predict_dy_error.ix['1977-10-01':].plot(ax=ax, style='r', label='Dynamic forecast (1978)')
        ci = self.predict_dy_ci.ix['1977-10-01':].copy()
        ci.iloc[:,0] -= self.endog.loc['1977-10-01':]
        ci.iloc[:,1] -= self.endog.loc['1977-10-01':]
        ax.fill_between(ci.index, ci.ix[:,0], ci.ix[:,1], color='r', alpha=0.1)

        legend = ax.legend(loc='lower left');
        legend.get_frame().set_facecolor('w')
        plt.show()

def eg1_ARIMA_111():
    '''
    ARIMA(p,d,q) model on the U.S. Wholesale Price Index (WPI) dataset
    The order argument is a tuple of the form (AR specification, Integration order, MA specification)
    '''
    # Dataset
    wpi1 = requests.get('http://www.stata-press.com/data/r12/wpi1.dta').content
    data = pd.read_stata(BytesIO(wpi1))
    data.index = data.t

    # Fit the model
    model = sm.tsa.statespace.SARIMAX(data['wpi'], trend='c', order=(1,1,1))
    res = model.fit(disp=False)
    print(res.summary())
    return res.params["intercept"]/float(1-res.params["ar.L1"])

def eg2_seasonal_ARIMA():
    '''
    Adds an MA(4) term to the ARIMA(1,1,1) specification to allow for an additive seasonal effect
    The dataset is quarterly thus MA(4) makes an annual seasonal effect
    '''
    # Dataset
    wpi1 = requests.get('http://www.stata-press.com/data/r12/wpi1.dta').content
    data = pd.read_stata(BytesIO(wpi1))
    data.index = data.t
    data['ln_wpi'] = np.log(data['wpi'])
    data['D.ln_wpi'] = data['ln_wpi'].diff()

    # Plot data
    fig, axes = plt.subplots(1, 2, figsize=(15,4))
    # Levels; not stationary
    axes[0].plot(data.index._mpl_repr(), data['wpi'], '-')
    axes[0].set(title='US Wholesale Price Index')
    # Log difference; seems stationary
    axes[1].plot(data.index._mpl_repr(), data['D.ln_wpi'], '-')
    axes[1].hlines(0, data.index[0], data.index[-1], 'r')
    axes[1].set(title='US Wholesale Price Index - difference of logs')

    # Plot acf, pacf
    fig, axes = plt.subplots(1, 2, figsize=(15,4))
    fig = sm.graphics.tsa.plot_acf(data.ix[1:, 'D.ln_wpi'], lags=40, ax=axes[0])
    fig = sm.graphics.tsa.plot_pacf(data.ix[1:, 'D.ln_wpi'], lags=40, ax=axes[1])

    # Fit the model
    model = sm.tsa.statespace.SARIMAX(data['ln_wpi'], trend='c', order=(1,1,1))
    res = model.fit(disp=False)
    print(res.summary())

def eg3_SARIMA():
    '''
    ARIMA (p,d,q)×(P,D,Q)s

    The order argument is a tuple of the form (AR specification, Integration order, MA specification)
    The seasonal argument is a tuple of the form
    (Seasonal AR specification, Seasonal Integration order, Seasonal MA, Seasonal periodicity).
    '''
    # Dataset
    air2 = requests.get('http://www.stata-press.com/data/r12/air2.dta').content
    data = pd.read_stata(BytesIO(air2))
    data.index = pd.date_range(start=datetime(data.time[0], 1, 1), periods=len(data), freq='MS')
    data['lnair'] = np.log(data['air'])

    # Fit the model
    # If simple_differencing=True, then the time series provided as endog is literatlly differenced and
    # an ARMA model is fit to the resulting new time series
    model = sm.tsa.statespace.SARIMAX(data['lnair'], \
                                    order=(2,1,0), \
                                    seasonal_order=(1,1,0,12), \
                                    simple_differencing=True)
    res = model.fit(disp=False)
    print(res.summary())

def eg4_ARMAX():
    '''
    with an explanator variables(the X part of ARMAX)

    the first equation is just a linear regression
    the second equation just describes the process followed by the error component as SARIMA
    '''
    # Dataset
    friedman2 = requests.get('http://www.stata-press.com/data/r12/friedman2.dta').content
    data = pd.read_stata(BytesIO(friedman2))
    data.index = data.time

    # Variables
    endog = data.loc['1959':'1981', 'consump']
    exog = sm.add_constant(data.loc['1959':'1981', 'm2'])

    # Fit the model
    model = sm.tsa.statespace.SARIMAX(endog, exog, order=(1,0,1))
    res = model.fit(disp=False)
    print(res.summary())

def eg5_dynamic_forecasting():
    '''
    First, estimate the parameters using data that excludes the last few observations
    (using data from 1959 to 1978)

    Next, get results for the full dataset but using the estimated parameters previously.
    '''
    # Dataset
    friedman2 = requests.get('http://www.stata-press.com/data/r12/friedman2.dta').content
    raw = pd.read_stata(BytesIO(friedman2))
    raw.index = raw.time
    data = raw.loc[:'1981']

    # first stage
    forecating = Forecasting(data)
    forecating.get_variables()
    forecating.first_stage()

    # second stage
    forecating.second_stage_one_step_ahead()
    forecating.second_stage_dynamic()

    # plot forecating
    forecating.plot_forecasting()

    # plot forecasting error
    forecating.plot_forecasting_error()


if __name__=="__main__":

    # ARIMA(p,d,q) model on the U.S. Wholesale Price Index (WPI) dataset
    mean_of_process = eg1_ARIMA_111()

    # Adds an MA(4) term to the ARIMA(1,1,1) specification to allow for an additive seasonal effect
    eg2_seasonal_ARIMA()

    # ARIMA (p,d,q)×(P,D,Q)s
    eg3_SARIMA()

    # with an explanator variables(the X part of ARMAX)
    eg4_ARMAX()

    # Forecasting
    eg5_dynamic_forecasting()
