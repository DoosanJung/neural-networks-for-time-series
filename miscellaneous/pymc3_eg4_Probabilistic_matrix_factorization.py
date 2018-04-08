#!/usr/bin/env python
'''
Studying Probabilistic programming..
using "PyMC3" and "Edward"

I modified the code a bit
reference: https://github.com/GalvanizeOpenSource/probabilistic-programming-intro
Authors: Galvanize DSI
Version: 1.0.0

The model discussed in this analysis was developed by Ruslan Salakhutdinov and Andriy Mnih.
All of the code and supporting text, when not referenced, is the original work of Mack Sweeney.
(https://www.linkedin.com/in/macksweeney)

<Probabilistic matrix factorization for recommender system>
* The rating R are modeled as draws from a Gaussian distribution
* Precision, alpha, a fixed parameter, reflects the uncertainty of the estimation;
    the inverse of the variance
* small precision parameters help control the growth of our latent parameters

Data
-------------------------------
v1 Jester dataset (http://eigentaste.berkeley.edu/dataset/)
At this point in time, v1 contains over 4.1 million continuous ratings in the range [-10, 10]
of 100 jokes from 73,421 users. These ratings were collected between Apr. 1999 and May 2003.
In order to reduce the training time of the model for illustrative purposes,
1,000 users who have rated all 100 jokes will be selected randomly.

Model
-------------------------------
Baseline models: uniform random, global mean, mean of means
Model: probabilistic matrix factorization

Evaluation metric
-------------------------------
Root mean squared error (RMSE)
'''
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__=="__main__":
