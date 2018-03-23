#!/usr/bin/env python
"""
Weights and biases for concrete examples and my logger
"""
import sys
import logging
from logging.handlers import RotatingFileHandler
from logging import handlers

class NetworkConfig(object):
    """
    Known weights and biases for the example networks
    They are arbitrarily selected for the testcases

    Notes
    -----------
    This contains the following models
    (1) One(1) hidden layer network
        N hidden layers = 1
        M input size = 2
        The number of nodes in hidden layer = 3

    (2) Two(2) hidden layers network
        N hidden layers = 1
        M input size = 2
        The number of nodes in both hidden layers = 3

    (3) More than two(2) hidden layers network
        N hidden layers = n (>2)
        M input size = m (>2)
        The number of nodes in all hidden layers = 3
    """

    #
    # weights for the network
    #
    Weights = {
        'w_input_to_hidden'  : [[0.5, 0.9],
                               [0.4, 1.0],
                               [0.4, 0.7]],
        'w_hidden_to_hidden' : [[0.73, 0.17, 0.65],
                               [0.11, 0.46, 0.19],
                               [0.05, 0.68, 0.73]],
        'w_hidden_to_output' : [0.012, 0.022, 0.017]
    }

    #
    # biases for the network
    #
    Biases = {
        'b_input_to_hidden':[0.14, 0.04, -0.01],
        'b_hidden_to_hidden': [-0.01 , 0.03, 0.2],
        'b_hidden_to_output': -0.08
    }

    @staticmethod
    def one_hidden_layer():
        """
        Weights and biases for a neural network with one hidden layer
        """
        return {
            # input to hidden layer
            'w1': NetworkConfig.Weights['w_input_to_hidden'],
            'b1': NetworkConfig.Biases['b_input_to_hidden'],
            # hidden to output layer
            'w2': NetworkConfig.Weights['w_hidden_to_output'],
            'b2': NetworkConfig.Biases['b_hidden_to_output']
        }

    @staticmethod
    def two_hidden_layers():
        """
        Weights and biases for a neural network with one hidden layer
        """
        return {
            # input to hidden layer
            'w1': NetworkConfig.Weights['w_input_to_hidden'],
            'b1': NetworkConfig.Biases['b_input_to_hidden'],
            # hidden layer to another hidden layer
            'w2': NetworkConfig.Weights['w_hidden_to_hidden'],
            'b2': NetworkConfig.Biases['b_hidden_to_hidden'],
            # hidden to output layer
            'w3': NetworkConfig.Weights['w_hidden_to_output'],
            'b3': NetworkConfig.Biases['b_hidden_to_output']
        }

class LogConfig(object):
    """
    for logging both in file and stdout
    """
    @staticmethod
    def my_logger():
        logger = logging.getLogger('')
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(format)
            logger.addHandler(ch)

            fh = logging.FileHandler('network_test.log')
            fh.setFormatter(format)
            logger.addHandler(fh)
        return logger
