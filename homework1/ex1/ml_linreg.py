'''
Created on Mar 27, 2016

@author: Babis Kaidos
'''

import numpy

class ml_linreg(object):
    
    def __init__(self):
        '''
        Constructor
        '''
        
    def train(self, data, target, lamda):
        (rows,) = data.shape
        data = numpy.expand_dims(data, axis=1)
        data = numpy.insert(data, 0, numpy.ones(rows), axis=1)
        (rows, columns) = data.shape
        T = numpy.matmul(numpy.transpose(data), target)
        K = numpy.matmul(numpy.transpose(data), data) + lamda*numpy.identity(columns)
        self.w = numpy.linalg.solve(K, T)
        y = numpy.dot(data,self.w)
        self.beta = rows / ((y-target)**2).sum()
    
    def test(self, data):
        (rows,) = data.shape
        data = numpy.expand_dims(data, axis=1)
        data = numpy.insert(data, 0, numpy.ones(rows), axis=1)
        y = numpy.dot(data, self.w)
        var = (1/self.beta)*numpy.ones(rows)
        return (y, var)