from os import listdir
from os.path import isfile, join

import numpy
import pickle

def load_mnist(path):
    print('Reading MNIST dataset from storage')

    trainfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and 'train' in f]
    train_data, train_target = load_mnist_from_files(trainfiles)

    testfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and 'test' in f]
    test_data, test_target = load_mnist_from_files(testfiles)

    print('Finished reading MNIST')
    return train_data, train_target, test_data, test_target

def load_mnist_from_files(files):
    data = None
    target = None

    for file in files:
        print('Reading file {}'.format(file))
        temp_data = numpy.loadtxt(file)
        if (numpy.any(data) == None):
            data = temp_data
        else:
            data = numpy.append(data, temp_data, axis=0)

        rows, columns = temp_data.shape
        temp_target = numpy.zeros((rows, 10))
        temp_target[:, int(file[-5])] = numpy.ones(rows)
        if (numpy.any(target) == None):
            target = temp_target
        else:
            target = numpy.append(target, temp_target, axis=0)

        print('Read {} rows, {} columns from {}'.format(rows, columns, file))

    return data, target

def pickle_mnist(traindata, traintarget, testdata, testtarget):
    f=open('./mnist/trdata', mode='wb+')
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
    f.close()

def unpickle_mnist():
    fo = open('./mnist/trdata', 'rb')
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
    fo.close()
    return traindata, traintarget, testdata, testtarget