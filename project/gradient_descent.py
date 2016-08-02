import mnist
import cifar10
import numpy
import pickle
import math
import time
from collections import deque


def likelihood_function(W, T, Y, lam):
    '''Calculate the cost for the given W'''
    return numpy.sum(T * numpy.log(Y + numpy.finfo(float).eps)) - lam / 2 * numpy.sum(numpy.linalg.norm(W, axis=0) ** 2)


def softmax(W, X):
    ''' Calculate softmax function for A=XW'''
    Y = numpy.dot(X, W.T)
    Y = Y - numpy.amax(Y, axis=1).reshape(Y.shape[0], 1)
    Y = numpy.exp(Y)
    Y = Y / numpy.sum(Y, axis=1).reshape(Y.shape[0], 1)
    return Y


def gradient_descent(X, T, step, iterations, precision):
    ''' Gradient descent to find optimal weights given a training set X,T
        We use momentum to accelerate the convergence'''

    def init_weights():
        numpy.random.seed(5)
        return deque((numpy.random.randn(T.shape[1], X.shape[1]), None, None))

    def init_momentum():
        return deque((0, None, None))

    def init_cost():
        return deque((-numpy.inf, -numpy.inf, -numpy.inf))

    # Initialize parameters
    W = init_weights()
    momentum = init_momentum()
    cost = init_cost()

    time_elapsed = 0
    for i in range(iterations):
        start = time.time()
        print('Starting iteration {}'.format(i))

        # Calculate the output for the current weights
        Y = softmax(W[0], X)

        # Calculate the cost function for these outputs
        cost.rotate(1)
        cost[0] = likelihood_function(W[0], T, Y, 0.05)

        # If the cost function has decreased increase the learning rate. Otherwise we moved over the "valey", so step
        # back to the previous iteration and cut the learning rate in half. Also if the cost difference between the
        # last 3 iterations is less than the precision we have converged and we should stop
        cost_diff = cost[0] - cost[1]
        if (cost_diff >= 0):
            if (cost_diff > 0 and cost_diff < precision and (cost[1] - cost[2] < precision)):
                break
            else:
                step = step * 1.05
        else:
            step = step * 0.5
            W.rotate(-1)
            cost.rotate(-1)
            momentum.rotate(-1)
            print('Stepping back: {}'.format(cost[0]))
            continue

        # Calculate the gradient
        gradient = numpy.dot(numpy.transpose(T - Y), X) - (0.05 * W[0])

        # Calculate the momentum. Momentum makes changing direction when descenting harder. This makes the algorithm
        # bump less on the "walls" of a "valey" when descenting.
        momentum.rotate(1)
        momentum[0] = 0.5 * momentum[1] + (step * gradient)

        # Adjust the weights according to the momentum calculated
        W.rotate(1)
        W[0] = W[1] + (step * gradient)

        time_elapsed += (time.time() - start)

        print('Gradient norm: {}, Likelihood: {}, step: {}'.format(numpy.linalg.norm(gradient, ord=2), cost[0], step))
        print('Elapsed: {:.3f}s, Remaining: {:.3f}s'.format(time_elapsed,
                                                            (time_elapsed / (i + 1)) * (iterations - i + 1)))

        # If we did something numerically wrong, then NaNs will appear. In this case we should stop
        if (numpy.isnan(numpy.min(W[0]))):
            print('NaNs appear')
            break

    return W[0]

def fuzzy_target(traintarget):
    ''' Transform the indicators from 0 and 1 to values close to 0 and 1
    This seems to work better for some algorithms'''
    for row in range(traintarget.shape[0]):
        for col in range(traintarget.shape[1]):
            if (traintarget[row, col] == 0):
                traintarget[row, col] = 0.0001
            else:
                traintarget[row, col] = 0.9991
    return traintarget


def main():
    #traindata, traintarget, testdata, testtarget = cifar10.load_cifar('./cifar-10-batches-py')
    #traindata, traintarget, testdata, testtarget = mnist.load_mnist('./mnist')
    traindata, traintarget, testdata, testtarget = mnist.unpickle_mnist()

    traintarget = fuzzy_target(traintarget)

    W = gradient_descent(traindata / 255, traintarget, 0.00001, 2000, 0.00001)
    Y = softmax(W, testdata / 255)
    res = numpy.argmax(Y, axis=1) - numpy.argmax(testtarget, axis=1)
    zeros = 0
    for row in range(res.shape[0]):
        if (res[row] == 0):
            zeros = zeros + 1
    print('Predicted correct {} out of 10000'.format(zeros))


if __name__ == '__main__':
    main()
