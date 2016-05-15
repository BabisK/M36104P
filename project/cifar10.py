from os import listdir
from os.path import isfile, join

import pickle
import numpy

def load_cifar(path):
    print('Reading CIFAR-10 dataset from storage')

    dicts = []

    trainfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and 'data_batch' in f]
    for f in trainfiles:
        dicts.append(unpickle(f))

    data = numpy.vstack(tuple([x['data'] for x in dicts]))
    ones = numpy.ones((data.shape[0], 1))
    data = numpy.hstack((ones,data))

    temp_target = numpy.hstack(tuple([x['labels'] for x in dicts]))
    target = numpy.zeros((temp_target.shape[0], 10))
    for r in range(temp_target.shape[0]):
        target[r, temp_target[r]] = 1

    test_data = unpickle(join(path, 'test_batch'))
    test_target = numpy.zeros((len(test_data['labels']), 10))
    for r in range(test_target.shape[0]):
        test_target[r, test_data['labels'][r]] = 1
    test_data = test_data['data']
    ones = numpy.ones((test_data.shape[0], 1))
    test_data = numpy.hstack((ones, test_data))

    return data, target, test_data, test_target

def unpickle(file):
    fo = open(file, 'rb')
    dictionary = pickle.load(fo, encoding='latin1')
    fo.close()
    return dictionary
