#!/usr/bin/env python
"""
Test cases: finding x_i
"""
import unittest
import numpy as np
from Network import Network
from Config import NetworkConfig, LogConfig

logger = LogConfig.my_logger()

class NetworkTestCase(unittest.TestCase):
    """
    Test cases:
    (1) CASE 1: Testing forward propagation
                1 hidden layer, 2 inputs, T == 0.5
                Desired output == "Bad"

    (2) CASE 2: Testing forward propagation
                2 hidden layers, 2 inputs, T == 0.5
                Desired output == "Bad"

    (3) CASE 3: Testing finding x_i values
                1 hidden layer, 2 inputs, T == 0.5
                Desired output == x_i value, "Good" for both elements in input
                x_0 (x_1 is constant)
                and x_1 (x_0 is constant)

    (4) CASE 4: Testing finding x_i values
                2 hidden layers, 2 inputs, T == 0.5
                Desired output == x_i value, "Good" for both elements in input
                x_0 (x_1 is constant)
                and x_1 (x_0 is constant)

    (5) CASE 5: Testing finding x_i values
                4 hidden layers, 4 inputs, T == 0.5
                Desired output == x_i value, "Good" for all elements in input
                x_0 (x_1, x_2, x_3 are constant),
                x_1 (x_0, x_2, x_3 are constant),
                x_2 (x_0, x_1, x_3 are constant),
                and x_3 (x_0, x_1, x_2 are constant)
    """
    def test_1_forward_propagate_one_hidden(self):
        logger.info("--"*20)
        logger.info("CASE1: Testing forward propagate case (one hidden layer)..")
        logger.info("--"*20)
        good_or_bad = None
        try:
            network = Network(n_hidden_layers=1, m_inputs=2, T=0.5)
            inputs = [1.4, -0.9] # arbitrarily selected inputs
            y = network.forward_propagate(inputs)
            logger.info("threshold value T: {}".format(0.5))
            good_or_bad = network.classify(y)
        except:
            logger.error("Failed test case - {}".format(self.test_1_forward_propagate_one_hidden.__name__))
            raise
        self.assertEqual("Bad", good_or_bad)

    def test_2_forward_propagate_two_hiddens(self):
        logger.info("--"*20)
        logger.info("CASE2: Testing forward propagate case (two hidden layers)..")
        logger.info("--"*20)
        good_or_bad = None
        try:
            network = Network(n_hidden_layers=2, m_inputs=2, T=0.5)
            inputs = [1.4, -0.9] # arbitrarily selected inputs
            y = network.forward_propagate(inputs)
            logger.info("threshold value T: {}".format(0.5))
            good_or_bad = network.classify(y)
        except:
            logger.error("Failed test case - {}".format(self.test_2_forward_propagate_two_hiddens.__name__))
            raise
        self.assertEqual("Bad", good_or_bad)

    def test_3_find_xi_one_hidden(self):
        logger.info("--"*20)
        logger.info("CASE3: Testing find x_i case (one hidden layer)..")
        logger.info("--"*20)
        good_or_bad = None
        try:
            network = Network(n_hidden_layers=1, m_inputs=2, T=0.5)
            inputs = [1.4, -0.9] # arbitrarily selected inputs
            y = network.forward_propagate(inputs)
            logger.info("threshold value T: {}".format(0.5))
            good_or_bad = network.classify(y)
            self.assertEqual("Bad", good_or_bad)

            good_or_bad = self.__finding_xi(network, inputs)
            self.assertEqual("Good", good_or_bad)
        except:
            logger.error("Failed test case - {}".format(self.test_3_find_xi_one_hidden.__name__))
            raise

    def test_4_find_xi_two_hiddens(self):
        logger.info("--"*20)
        logger.info("CASE4: Testing find x_i case (two hidden layers)..")
        logger.info("--"*20)
        good_or_bad = None
        try:
            network = Network(n_hidden_layers=2, m_inputs=2, T=0.5)
            inputs = [1.4, -0.9] # arbitrarily selected inputs
            y = network.forward_propagate(inputs)
            logger.info("threshold value T: {}".format(0.5))
            good_or_bad = network.classify(y)
            self.assertEqual("Bad", good_or_bad)

            good_or_bad = self.__finding_xi(network, inputs)
            self.assertEqual("Good", good_or_bad)
        except:
            logger.error("Failed test case - {}".format(self.test_4_find_xi_two_hiddens.__name__))
            raise

    def test_5_find_xi_N_hiddens(self):
        logger.info("--"*20)
        logger.info("CASE5: Testing forward propagate case (N hidden layers)..")
        logger.info("--"*20)
        result = None
        try:
            network = Network(n_hidden_layers=4, m_inputs=4, T=0.5)
            inputs = [1.4, -0.9, 0.8, 2.3] # arbitrarily selected inputs
            y = network.forward_propagate(inputs)
            logger.info("threshold value T: {}".format(0.5))
            good_or_bad = network.classify(y)
            logger.info("Since the weights and biases are randomly generated, couldn't tell good or bad beforehand..")
            # find x_i only if the output was 'Bad'
            if good_or_bad == "Bad":
                good_or_bad = self.__finding_xi(network, inputs)
                # after finding x_i, it can still be 'bad' when the y value did not converge to the threshold value T
                if good_or_bad == "Bad":
                    logger.error("y value did not converge to the threshold value T")
                    logger.error("y value: {}".format(y[0]))
                    logger.error("threshold T: {}".format(network.T))
            else:
                logger.info("The ouput was 'Good' in the first place..")
            self.assertEqual("Good", good_or_bad)
        except:
            logger.error("Failed test case - {}".format(self.test_5_find_xi_N_hiddens.__name__))
            raise

    def __finding_xi(self, network, inputs):
        for i in xrange(len(inputs)):
            logger.info(" ")
            logger.info("Changing the element no.{} of the inputs".format(i+1))
            result = network.find_xi(inputs, i)
            logger.info("The result for x_i is {}".format(result))
            logger.info("Let's check whether it changes the state of the output!! ")
            good_or_bad = network.test_solution(result, i)
        return good_or_bad

if __name__=="__main__":
    unittest.main()
