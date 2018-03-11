#!/usr/bin/env python
'''
Studying Probabilistic programming..
using "PyMC3" and "Edward"

I modified the code a bit
reference: https://github.com/GalvanizeOpenSource/probabilistic-programming-intro
Authors: Galvanize DSI
Version: 1.0.0

This example originally comes from Cam Davidson-Pilon's book "Bayesian Methods for Hackers"
https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
'''
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt

class SwitchPointExample(object):
    """
    We will be looking at count data,
    specifically at the frequency of text messages recieved over a period of time.

    We will use the Poisson distribution to help use investigate this series of events.
    Text-message count at day i :Ci
    Ci ~ Poisson(lambda)
    """
    def __init__(self, filename):
        self.count_data = np.loadtxt(filename)
        self.n_count_data = len(self.count_data)

    def plot_data(self, show=True):
        print("plotting the data...")
        fig = plt.figure(figsize=(12.5, 3.5))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(self.n_count_data), self.count_data, color="#348ABD")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("count of text-msgs received")
        ax.set_title("Did the user's texting habits change over time?")
        ax.set_xlim(0, self.n_count_data);
        if show == True:
            plt.show()

    def infer_lambda(self):
        """
        Ci ~ Poisson(lambda)

        Is there a day ("tau") where the lambda suddenly jumps to a higher value?
        We are looking for a 'switchpoint' s.t. lambda
            (1) (lambda_1 if t < tau) and (lambda_2 if t > tau)
            (2) lambda_2 > lambda_1

        lambda_1 ~ Exponential(alpha)
        lambda_2 ~ Exponential(alpha)

        tau ~ Discrete_uniform(1/n_count_data)
        """
        print("Infer with PyMC3...")
        with pm.Model() as model:
            ## assign lambdas and tau to stochastic variables
            alpha = 1.0/self.count_data.mean()
            lambda_1 = pm.Exponential("lambda_1", alpha)
            lambda_2 = pm.Exponential("lambda_2", alpha)
            tau = pm.DiscreteUniform("tau", lower=0, upper=self.n_count_data)

            ## create a combined function for lambda (it is still a random variable)
            idx = np.arange(self.n_count_data) # Index
            lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)

            ## combine the data with our proposed data generation scheme
            observation = pm.Poisson("obs", lambda_, observed=self.count_data)

            ## inference
            step = pm.Metropolis()
            self.trace = pm.sample(10000, tune=5000,step=step)

            ## get the variables we want to plot from our trace
            self.lambda_1_samples = self.trace['lambda_1']
            self.lambda_2_samples = self.trace['lambda_2']
            self.tau_samples = self.trace['tau']

    def plot_hist(self,show=True):
        print("Plotting histograms of the result")
        # draw histogram of the samples:
        fig = plt.figure(figsize=(12.5,10))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        for ax in [ax1,ax2]:
            ax.set_autoscaley_on(False)

        ## axis 1
        ax1.hist(self.lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
                 label="posterior of $\lambda_1$", color="#A60628", normed=True)
        ax1.legend(loc="upper left")
        ax1.set_title(r"""Posterior distributions of the variables
            $\lambda_1,\;\lambda_2,\;\tau$""")
        ax1.set_xlim([15, 30])
        ax1.set_xlabel("$\lambda_1$ value")

        ## axis 2
        ax2.hist(self.lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
                 label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
        ax2.legend(loc="upper left")
        ax2.set_xlim([15, 30])
        ax2.set_xlabel("$\lambda_2$ value")

        ## axis 3
        w = 1.0 / self.tau_samples.shape[0] * np.ones_like(self.tau_samples)
        ax3.hist(self.tau_samples, bins=self.n_count_data, alpha=1,
                 label=r"posterior of $\tau$",
                 color="#467821", weights=w, rwidth=2.)
        ax3.set_xticks(np.arange(self.n_count_data))

        ax3.legend(loc="upper left")
        ax3.set_ylim([0, .75])
        ax3.set_xlim([35, self.n_count_data-20])
        ax3.set_xlabel(r"$\tau$ (in days)")
        ax3.set_ylabel("probability");
        plt.subplots_adjust(hspace=0.4)
        if show == True:
            plt.show()

    def plot_result(self,show=True):
        print("Plotting expected number of text-messages received")
        fig = plt.figure(figsize=(12.5,5))
        ax = fig.add_subplot(111)

        N = self.tau_samples.shape[0]
        expected_texts_per_day = np.zeros(self.n_count_data)
        for day in range(0, self.n_count_data):
            ix = day < self.tau_samples
            expected_texts_per_day[day] = (self.lambda_1_samples[ix].sum()
                                           + self.lambda_2_samples[~ix].sum()) / N

        ax.plot(range(self.n_count_data), expected_texts_per_day, lw=4, color="#E24A33",
                 label="expected number of text-messages received")
        ax.set_xlim(0, self.n_count_data)
        ax.set_xlabel("Day")
        ax.set_ylabel("Expected # text-messages")
        ax.set_title("Expected number of text-messages received")
        ax.set_ylim(0, 60)
        ax.bar(np.arange(self.n_count_data), self.count_data, color="#348ABD", alpha=0.65,label="observed texts per day")
        ax.legend(loc="upper left");
        if show==True:
            plt.show()

if __name__=="__main__":
    print(" ")
    switchpoint = SwitchPointExample(filename="miscellaneous/data/txtdata.csv")
    switchpoint.plot_data(show=True)
    print(" ")

    switchpoint.infer_lambda()
    print(" ")

    switchpoint.plot_hist(show=True)
    print(" ")

    switchpoint.plot_result()
    print(" ")
