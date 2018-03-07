#!/usr/bin/env python
'''
'''
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime

class RunOLS(object):
    def __init__(self, df, order, **kwargs):
        self.df = df
        self.order = order
        self.seasonal = kwargs.pop("seasonal_component", False)
        if kwargs:
            raise TypeError("Unexpected **kwargs: {}".format(kwargs))

    def __add_polynomial(self, x_col):
        '''
        Not a great practice to add polynomials: can be overfitting it
        '''
        if self.order == 1:
            self.df['{}^{}'.format(x_col, 1)] = self.df[x_col]
        else:
            for i in xrange(1, self.order+1):
                self.df['{}^{}'.format(x_col, i)] = self.df[x_col]**i
        return self.df

    def run_ols(self, y_col, x_col):
        self.df = self.__add_polynomial(x_col)
        try:
            y = self.df[y_col].values
            x_col = ['time^{}'.format(i) for i in xrange(1, self.order+1)]
            if self.seasonal:
                x_col.append(self.seasonal)
                x_col = pd.get_dummies(self.df[x_col], columns=[self.seasonal])
                X = sm.add_constant(x_col)
            else:
                X = sm.add_constant(self.df[x_col])
            self.model = sm.OLS(y,X).fit()
            ts = pd.Series(self.df[y_col])
            ts.plot(figsize=(12,8))
            dates = pd.date_range('1980-01','2011-01', freq='M')
            pd.Series(self.model.fittedvalues, index=dates).plot()
            plt.show()
            print self.model.summary()
        except:
            print("Failed to run OLS")
            raise

def get_monthly_data():
    # monthly data
    df = pd.read_csv('data/birth.txt', delim_whitespace=True)

    # from January 1980 and ending December 2010
    df['time']= range(1,df.shape[0]+1)
    dates = pd.date_range('1980-01','2011-01', freq='M')

    df['dates'] = dates
    df=df.set_index('dates')
    df['months'] = pd.DatetimeIndex(df.index).month
    df['years'] = pd.DatetimeIndex(df.index).year
    return df

def EDA(df, **kwargs):
    '''
    You need to provide starting_year and ending_year
    for plot_zoom_in
    '''
    show_average = kwargs.pop("show_average", False)
    plot_overall_data = kwargs.pop("plot_overall_data", False)
    plot_zoom_in = kwargs.pop("plot_zoom_in", False)
    superimpose = kwargs.pop("superimpose", False)
    if kwargs:
        raise TypeError("Unexpected **kwargs: {}".format(kwargs))

    if show_average == True:
        # Monthly average
        print("Montly average number of births")
        print(df.groupby(['months'])['num_births'].mean())

        # Yearly average
        print("Yearly average number of births")
        print(df.groupby(['years'])['num_births'].mean())

    if plot_overall_data == True:
        # Plotting the overall data
        ts = pd.Series(df['num_births'])
        ts.plot(figsize=(12,8), title="overall_data")
        plt.show()

    if plot_zoom_in == True:
        # Plotting the data from 2008-2010
        # to see seasonal pattern more apparently
        ts = pd.Series(df['num_births'])
        starting_year = raw_input("Starting_year: ")
        ending_year = raw_input("Ending_year: ")
        if datetime.strptime(starting_year, "%Y") < datetime.strptime("1980", "%Y"):
            raise Exception("starting_year >= 1980")
        if datetime.strptime(ending_year, "%Y") > datetime.strptime("2011", "%Y"):
            raise Exception("ending_year <= 2010")
        ts[str(starting_year):str(ending_year)].plot(figsize=(12,8), title="zoom_in")
        plt.show()

    if superimpose == True:
        # get quarterly means that follow the seasons of the year \
        # (spring, summer, fall, winter)
        ts = pd.Series(df['num_births'])
        ts.plot(figsize=(12,8),title="Superimpose the yearly averages and \
                                the seasonal averages onto the monthly data")
        ts.resample('A').mean().plot()
        ts.resample('Q-NOV').mean().plot()
        plt.show()

def residual_analysis(model):
    '''
    Take the best models and see the residuals
    '''
    df = get_monthly_data()
    df['resids'] = model.resid
    df.plot(kind='scatter', x='time', y='resids', figsize=(12,8))
    plt.show()


if __name__=="__main__":
    df = get_monthly_data()

    # you need to provide starting_year and ending_year for plot_zoom_in
    EDA(df, show_average = True, plot_overall_data = True, plot_zoom_in = True)
    EDA(df, superimpose = True)

    # fit the overall trend with increasing polynomial terms
    # compare AIC or BIC
    aics = {}
    bics = {}
    for n in xrange(1,4):
        ols = RunOLS(get_monthly_data(), order=n)
        ols.run_ols(y_col='num_births', x_col='time')
        aics['order_{}'.format(n)] = ols.model.aic
        bics['order_{}'.format(n)] = ols.model.bic
    print "AICs with additional orders: ", aics
    print "BICs with additional orders: ", bics

    # add seasonal_component to the model
    ols = RunOLS(get_monthly_data(), order=3, seasonal_component='months')
    ols.run_ols(y_col='num_births', x_col='time')
    print "AIC with seasonal component: ", ols.model.aic
    print "BIC with seasonal component: ", ols.model.bic

    # residual analysis: Is there an obvious pattern of residuals w.r.t to time?
    residual_analysis(ols.model)
