import mnist
import cifar10
import numpy
import pickle
import math

def likelihood_function(W, T, Y, lam):
    return numpy.sum(T*numpy.log(Y+numpy.finfo(float).eps)) - lam/2*numpy.sum(numpy.linalg.norm(W, axis=0))

def softmax(W, X):
    Y = numpy.dot(X, W.T)
    for row in range(Y.shape[0]):
        Y[row, :] = Y[row, :] - max(Y[row,:])
        Y[row, :] = numpy.exp(Y[row, :]) / sum(numpy.exp(Y[row, :]))

    return Y


def gradient_descent(X, T, step, iterations, precision):
    numpy.random.seed(1)
    W = numpy.random.rand(T.shape[1], X.shape[1])
    W_old = [None, None]
    v_old = [0, 0]
    cost_old = [-numpy.inf, -numpy.inf]
    for i in range(iterations):
        print('Starting iteration {}'.format(i))
        Y = softmax(W, X)
        cost = likelihood_function(W, T, Y, 0.05)
        if (cost > cost_old[0]):
            step = step * 1.1
            cost_old[1] = cost_old[0]
            cost_old[0] = cost
        else:
            step = step * 0.5
            W = W_old[0]
            W_old[0] = W_old[1]
            cost = cost_old[0]
            cost_old[0] = cost_old[1]
            print('Stepping back')
            continue
        gradient = numpy.dot(numpy.transpose(T - Y), X) - (0.05 * W)
        v = 0.5*v_old[0] + (step * gradient)
        v_old[1] = v_old[0]
        v_old[0] = v
        W_old[1] = W_old[0]
        W_old[0] = W
        W = W + v
        print('Gradient norm: {}, Likelihood: {}, step: {}'.format(numpy.linalg.norm(gradient, ord=2), cost, step))
        if(numpy.isnan(numpy.min(W))):
            print('NaNs appear')
            break
        if(numpy.linalg.norm(W_old[0] - W, ord=2) < precision):
            print('Converged')
            break

    return W

def main():
    traindata, traintarget, testdata, testtarget = cifar10.load_cifar('./cifar-10-batches-py')
    #traindata, traintarget, testdata, testtarget = mnist.load_mnist('./mnist')
    '''f=open('./mnist/trdata', mode='wb+')
    pickle.dump(traindata, f)
    f.close()
    f = open('./mnist/trtarget', mode='wb+')
    pickle.dump(traintarget, f)
    f.close()
    f = open('./mnist/tedata', mode='wb+')
    pickle.dump(testdata, f)
    f.close
    f = open('./mnist/tetarget', mode='wb+')
    pickle.dump(testtarget, f)
    f.close()'''
    '''fo = open('./mnist/trdata', 'rb')
    traindata = pickle.load(fo)
    fo.close()
    fo = open('./mnist/trtarget', 'rb')
    traintarget = pickle.load(fo)
    fo.close()
    fo = open('./mnist/tedata', 'rb')
    testdata = pickle.load(fo)
    fo.close()
    fo = open('./mnist/tetarget', 'rb')
    testtarget = pickle.load(fo)
    fo.close()'''

    for row in range(traintarget.shape[0]):
        for col in range(traintarget.shape[1]):
            if(traintarget[row,col] == 0):
                traintarget[row, col] = 0.01
            else:
                traintarget[row, col] = 0.91

    W = gradient_descent(traindata/255, traintarget, 0.00001, 1000, 0.00001)
    Y = softmax(W, testdata/255)
    for r in range(Y.shape[0]):
        m = max(Y[r,:])
        for c in range(Y.shape[1]):
            if(Y[r,c] == m):
                Y[r,c] = 1
            else:
                Y[r,c] = 0
    res = Y - testtarget
    zeros = 0
    for row in range(res.shape[0]):
        if(res[row, :].min() == 0 and res[row, :].max() == 0):
            zeros = zeros +1
    print(zeros)

if __name__ == '__main__':
    main()
