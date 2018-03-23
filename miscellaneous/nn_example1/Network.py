#!/usr/bin/env python
"""
A feed forward neural network examples
"""
import numpy as np
from scipy import optimize
from Config import NetworkConfig, LogConfig

logger = LogConfig.my_logger()

class Network(object):
    """
    A feed forward neural network example with
        N hidden layers,
        M inputs,
        1 output neuron,
        and all the weights of the Network(W) are known.

    Activation functions used here are as follows:
        ReLU: activatoin function for all hidden layers
        sigmoid function: activation function for the ouput neuron - Here I used
                          f(x) = 1 / (1+ exp(-x)) (logistic function)

    Parameters
    -----------
    n_hidden_layers: the number of hidden layers (assume positive integer)
    m_inputs: size of the input (assume positive integer)
    T: some known threshold (assume 0 <= T <= 1)

    Assumptions
    -----------
    Sigmoid function: I used f(x) = 1 / (1+ exp(-x)) (logistic function) because
                     it is simple and it works as a valid sigmoid function

    Threshold T: Since I used logistic function, I also assume that 0 <= T <= 1.
                If T < 0, then all the output will be classified as 'good'
                or if T > 1, then all the output will be classified as 'bad'

    The number of nodes in hidden layers: For simplicity, all the hidden layers
                have same number of nodes: an arbitrary positive integer, e.g. 3

    Notes
    -----------
    Following test cases included in Network_test.py:

        usage: $python Network_test.py

    (1) One(1) hidden layer network (CASE1, CASE3)
        N hidden layers = 1
        M input size = 2
        H the number of nodes in hidden layer = 3

    (2) Two(2) hidden layers network (CASE2, CASE4)
        N hidden layers = 2
        M input size = 2
        H the number of nodes in both hidden layers = 3

    (3) More than two(2) hidden layers network (CASE 5)
        N hidden layers = n (>2)
        M input size = m (>2)
        The number of nodes in all hidden layers = 3
        The weigths and biases for this network will be randomly generated
    """

    def __init__(self, n_hidden_layers, m_inputs, T):
        """
        If n equals to 1, then we have a network with one hidden layer
        Else if n equalts to 2, then we have a network with two hidden layers
        Else if n is greater than 2, then we have a network with more than two hidden layers
        """
        self.n = self.__check_positive_integer(n_hidden_layers)
        self.m = self.__check_positive_integer(m_inputs)
        self.T = self.__bounded_threshold(T)

        if self.n == 1:
            self.m = 2
            self.network_config = NetworkConfig.one_hidden_layer()
            logger.info("Weights and biases will be fetched for one hidden layer network")

        elif self.n == 2:
            self.m = 2
            self.network_config = NetworkConfig.two_hidden_layers()
            logger.info("Weights and biases will be fetched for two hidden layers network")

        elif self.n > 2:
            logger.info("weigths and biases will be randomly generated for this example")
            # I used plain Python dictionary to store weights and biases
            self.w = {}
            self.b = {}

            # input to hidden layer
            self.w[0] = np.random.randn(3,self.m)
            self.b[0] = np.random.randn(3,)

            # hidden layer to next hidden layer
            for i in xrange(self.n - 1):
                self.w[i+1] = np.random.randn(3,3)
                self.b[i+1] = np.random.randn(3,)

            # hidden layer to output layer
            self.w[self.n] = np.random.randn(3,)
            self.b[self.n] = np.random.randn(1,)
        else:
            raise Exception("Please choose positive integer for the number of hidden layer(s).\
                        you chose {}".format(self.n))

    def relu(self, z):
        """
        The activation function for hidden layer(s)
        """
        return np.maximum(z, 0)

    def sigmoid(self,z):
        """
        The activation function for output layer
        Here I used f(x) = 1 / (1+ exp(-x)) (logistic function)
        """
        return 1/(1+np.exp(-z))

    def add_hidden(self, w, a, b):
        """
        Adding hidden layer to network
        It uses ReLU as an activation function
        """
        z = np.dot(w, a) + b
        a = self.relu(z)
        return a

    def add_output(self, w, a, b):
        """
        Adding output layer to network
        It uses sigmoid function as an activation function
        """
        z = np.dot(w, a) + b
        y = self.sigmoid(z)
        return y

    def forward_propagate(self, inputs):
        """
        calculate the output by propagating inputs through each layer
        until output layer returns its value
        """
        if self.n==1:
            logger.info("inputs value: {}".format(inputs))

            a1 = self.add_hidden(self.network_config['w1'], inputs, self.network_config['b1'])
            logger.info("a1 value: {}".format(a1))

            y = self.add_output(self.network_config['w2'], a1, self.network_config['b2'])
            logger.info("y value: {}".format(y))

        elif self.n ==2:
            logger.info("inputs value: {}".format(inputs))

            a1 = self.add_hidden(self.network_config['w1'], inputs, self.network_config['b1'])
            logger.info("a1 value: {}".format(a1))

            a2 = self.add_hidden(self.network_config['w2'], a1, self.network_config['b2'])
            logger.info("a2 value: {}".format(a2))

            y = self.add_output(self.network_config['w3'], a2, self.network_config['b3'])
            logger.info("y value: {}".format(y))

        elif self.n > 2:
            logger.info("inputs value: {}".format(inputs))

            a1 = self.add_hidden(self.w[0], inputs, self.b[0])
            a1 = a1.flatten()
            logger.info("a1 value: {}".format(a1))

            for i in xrange(self.n - 1):
                a1 = self.add_hidden(self.w[i+1], a1, self.b[i+1])
                a1 = a1.flatten()
                logger.info("a{0} value: {1}".format(i+2, a1))

            y = self.add_output(self.w[self.n], a1, self.b[self.n])
            logger.info("y value: {}".format(y))

        else:
            raise Exception("Please choose positive integer for the number of hidden layer(s).\
                                    you chose {}".format(self.n))
        return y

    def classify(self, y):
        """
        If the output value is greater than threshold T, then it will be classified as 'good'
        Otherwise, it will be classified as 'bad'
        """
        if (y <= self.T):
            logger.info("output y value <= threshold T")
            logger.info("Classified as Bad")
            return "Bad"
        else:
            logger.info("output y value > threshold T")
            logger.info("Classified as Good")
            return "Good"

    def __objective(self, x0):
        """
        It is an objectvie function for Scipy's fsolve
        """
        inputs = [x0 if x==self.initial_guess else x for x in self.inputs]
        LHS = self.T
        RHS = self.forward_propagate(inputs)
        return LHS - RHS

    def find_xi(self, inputs, i):
        """
        Scipy's fsolve returns the roots of f(x) = 0
        """
        self.inputs = inputs
        self.initial_guess = self.inputs[i]
        result = optimize.fsolve(self.__objective, self.initial_guess)
        return result

    def test_solution(self, result, i):
        """
        To test the x_i can change the classification.
        Since my x_i is the point which values from output value equals to threshold,
        small number has been added to the result in order to see the x_i is indeed the point
        """
        tmp = self.inputs.pop(i)

        # test with very small increase in inputs
        add = pow(10,-10)
        logger.info("adding this amount {} to the result".format(add))
        self.inputs.insert(i, list(result + add)[0])
        y = self.forward_propagate(self.inputs)
        good_or_bad = self.classify(y)

        # restore the inputs
        self.inputs.pop(i)
        self.inputs.insert(i, tmp)
        return good_or_bad

    def __bounded_threshold(self, T):
        """
        Here I assumed that 0 <= T <= 1 because I used f(x) = 1 / (1+ exp(-x))
        """
        if not ((0<=T) and (T<=1)):
            raise ValueError("0 <= T <= 1")
        return T

    def __check_positive_integer(self, x):
        """
        Here I assumed that both the number of inputs M and the number of hidden
        layers are positive integer
        """
        if not (isinstance(x, (int, long)) and x>0):
            raise ValueError("{} should be a positive integer".format(x))
        return x
