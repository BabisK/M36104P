'''
Created on Apr 2, 2016

@author: babis
'''

import numpy
from scipy.stats import logistic

class ml_logreg(object):

    def __init__(self):
        '''
        Constructor
        '''
        
    def train(self, data, target, lamda, iterations, tolerance, learning_rate):
        (rows, columns) = data.shape
        data = numpy.insert(data, 0, numpy.ones(rows), axis=1)
        (rows, columns) = data.shape
        
        self.w = numpy.zeros(columns)
        
        ew_old = -numpy.inf
        for i in range(iterations):
            yx = numpy.dot(data, self.w)
            s = logistic.cdf(yx)
            
            ew = (target*numpy.log(s) + (1-target)*numpy.log(1-s)).sum() - (0.5*lamda)*(numpy.matmul(numpy.transpose(self.w), self.w))
            print('Iteration: {}, Cost function: {}'.format(i, ew));
            
            if abs(ew - ew_old) < tolerance:
                break;
            
            gradient = numpy.matmul(numpy.transpose(data), target-s) - lamda*self.w
            
            self.w = self.w + learning_rate*gradient
            ew_old = ew
            
    def test(self, data):
        (rows, columns) = data.shape
        data = numpy.insert(data, 0, numpy.ones(rows), axis=1)
        (rows, columns) = data.shape
        y = logistic.cdf(numpy.dot(data,self.w))
        pred = numpy.around(y)
        
        return (y, pred)
        