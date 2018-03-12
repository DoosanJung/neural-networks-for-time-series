#!/usr/bin/env python
'''
Studying Probabilistic programming..
using "PyMC3" and "Edward"

I modified the code a bit
reference: https://github.com/GalvanizeOpenSource/probabilistic-programming-intro
Authors: Galvanize DSI
Version: 1.0.0

<Bayesian Linear Regression>
When the regression model has errors that have a normal distribution,
and if a particular form of prior distribution is assumed,
explicit results are available for the posterior probability distributions of the model's parameters.

In general, it may be impossible or impractical to derive the posterior distribution analytically.
However, it is possible to approximate the posterior by an approximate Bayesian inference method
such as Monte Carlo sampling

reference: https://en.wikipedia.org/wiki/Bayesian_linear_regression

<NUTS>
reference: https://arxiv.org/abs/1111.4246
"The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo",
Matthew D. Hoffman & Andrew Gelman
'''
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BayesianLinearRegression(object):
    """
    For a set of N data points (X, y) = {(xn, yn)},

        p(W) = Normal(w | 0, sigma(w)^2)
        p(b) = Normal(b | 0, sigma(b)^2)
        p(y |w, b, X)  = Normal(y1|x1T*w + b, sigma(y)^2)


    The latent variables are the linear model's weights w and intercetp b
    Assume the prior and likelihood variance are known: sigma(w)^2, sigma(b)^2, sigma(y)^2
    The mean of the likelihood is given by a linear transformation of the inputs in xn

    reference: Murphy, K. P.(2012), Machine Learning: A probabilistic perspective, MIT Press
    """
    def __init__(self, num_data_points, slope, intercept):
        self.n = num_data_points
        self._a = slope
        self._b = intercept

    def generate_data(self):
        np.random.seed(123)
        self.x = np.linspace(0, 1, self.n)
        self.y = self._a*self.x + self._b + np.random.randn(self.n)

    def plot_data(self, show=True):
        if show == True:
            print("plotting the data...")
            plt.plot(self.x, self.y)
            plt.title("gerenerated data")
            plt.show()

    def infer_with_pymc3(self, n_iteration):
        with pm.Model() as linreg:
            a = pm.Normal('a', mu=0, sd=100)
            b = pm.Normal('b', mu=0, sd=100)
            sigma = pm.HalfNormal('sigma', sd=1)
            # http://docs.pymc.io/api/distributions/continuous.html#pymc3.distributions.continuous.HalfNormal

            y_est = a*self.x + b
            likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=self.y)

            self.trace = pm.sample(n_iteration, random_seed=123)

    def plot_params(self, show=True):
        if show == True:
            print("plotting the parameters a, b...")
            pm.traceplot(self.trace, varnames=['a', 'b'])
            plt.show()

    def plot_regression_result(self, show=True):
        if show == True:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)

            ax.scatter(self.x, self.y, s=40, label='data')
            for a_, b_ in zip(self.trace['a'][-100:], self.trace['b'][-100:]):
                ax.plot(self.x, a_*self.x + b_, c='black', alpha=0.1)
            ax.plot(self.x, self._a*self.x + self._b, label='true regression line', lw=4., c='red')
            ax.legend(loc='best')
            plt.show()

if __name__=="__main__":
    bayesian_lr = BayesianLinearRegression(num_data_points=11, slope=6, intercept=2)

    print("Generating fake data")
    bayesian_lr.generate_data()
    print(" ")
    bayesian_lr.plot_data()
    print(" ")

    print("Infer with PyMC3...")
    bayesian_lr.infer_with_pymc3(n_iteration=10000)
    print(" ")
    bayesian_lr.plot_params()
    print(" ")
    bayesian_lr.plot_regression_result()
