#!/usr/bin/env python
'''
Studying Probabilistic programming..
using "PyMC3" and "Edward"

I modified the code a bit
reference: https://github.com/GalvanizeOpenSource/probabilistic-programming-intro
Authors: Galvanize DSI
Version: 1.0.0

Box's loop:
                 Data
                  |
                  V
    Model -> Infererence -> criticize
       <------------------------'

'''
import numpy as np
import pymc3 as pm
import scipy.stats as scs
import matplotlib.pyplot as plt

class CoinTossExample(object):
    """
    We are looking to determine if our coin is biased.
    We can investigate the fairness of a coin in several ways.
    One useful distribution is the Binomial.
    It is parameterized by the number of trials 'n'
    and the success probability in each trial 'p'
    """
    def __init__(self, n, true_p):
        """
            param n: sample data size
            param true_p: actual value of p for coin
        """
        self.n = n # e.g. 1000
        self.p = 0.5 # unbiased
        self.true_p = true_p # potentially biased

    def get_sample_data_size(self):
        print("sample data size: {}".format(self.n))

    def get_true_p(self):
        print("actual value of p for coin: {}".format(self.true_p))

class CoinTossData(CoinTossExample):
    def __init__(self, n, true_p):
        super(CoinTossData, self).__init__(n, true_p)
        self.results = scs.bernoulli.rvs(p=self.true_p, size=self.n)
        self.heads = sum(self.results)
        print("We observed %s heads out of %s"%(self.heads,self.n))

    def get_results(self):
        print("Realized result: {}".format(self.results))

    def get_true_p(self):
        print("The number of heads: {}".format(self.heads))

class CoinTossEDA(CoinTossData):
    def expected_distn_unbiased(self):
        rv = scs.binom(self.n, self.p)
        mu = rv.mean()
        sd = rv.std()
        print("The expected distribution for a fair coin is mu=%s, sd=%s"%(mu,sd))

    def p_value_by_simulation(self, n_simulation):
        xs = np.random.binomial(self.n, self.p, n_simulation) # e.g. 100000
        print("Simulation p-value: {}".format((2*np.sum(xs >= self.heads) / float(xs.size))))

    def p_value_by_test(self):
        print("Binomial test (p-value): {}".format(scs.binom_test(self.heads, self.n, self.p)))

    def mle(self):
        print("Maximum likelihood: {}".format((np.sum(self.results)/float(len(self.results)))))

    def bootstrap(self, n_samples): # e.g. 100000
        bs_samples = np.random.choice(self.results, (n_samples, len(self.results)), replace=True)
        bs_ps = np.mean(bs_samples, axis=1)
        bs_ps.sort()
        print("Bootstrap CI: (%.4f, %.4f)" % (bs_ps[int(0.025*n_samples)], bs_ps[int(0.975*n_samples)]))

class InferPyMC3(CoinTossData):

    def infer_with_pymc3(self):
        print("n = {}".format(self.n))
        print("heads = {}".format(self.heads))
        self.alpha = 2
        self.beta = 2

        n_iteration = 1000
        with pm.Model() as model: # basic_model
            # define priors
            p = pm.Beta('p', alpha=self.alpha, beta=self.beta)

            # define likelihood
            y = pm.Binomial('y', n=self.n, p=p, observed=self.heads)

            # inference
            start = pm.find_MAP() # Use MAP estimate (optimization) as the initial state for MCMC
            step = pm.Metropolis() # Have a choice of samplers
            self.trace = pm.sample(n_iteration, step, start, random_seed=123, progressbar=True)

    def plot(self, show=True):
        if show == True:
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)

            ax.hist(self.trace['p'], 15, histtype='step', normed=True, label='post');
            x = np.linspace(0, 1, 100)
            ax.plot(x, scs.beta.pdf(x, self.alpha, self.beta), label='prior');
            ax.legend(loc='best');
            plt.show()

if __name__=="__main__":
    print(" ")
    true_p = float(raw_input("Input actual success probability (0 <= probability <=1): "))
    if (true_p < 0) or (true_p > 1):
        raise Exception("0 <= true_p <= 1")

    print(" ")
    print("prepare the sample data with {}...".format(true_p))
    data = CoinTossEDA(n=1000, true_p = true_p)
    data.expected_distn_unbiased()
    data.p_value_by_simulation(n_simulation=100000)
    data.p_value_by_test()
    data.mle()
    data.bootstrap(n_samples=100000)
    print(" ")

    print("Infer with PyMC3...")
    infer = InferPyMC3(n=1000, true_p = true_p)
    infer.infer_with_pymc3()
    infer.plot()
