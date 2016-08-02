import numpy as np
import pickle
import mnist
import time

def softmax(A):
    A = A - np.amax(A, axis = 1).reshape(A.shape[0],1)
    A = np.exp(A)
    A = A / np.sum(A, axis=1).reshape(A.shape[0],1)
    return A

def sigmoid(A):
    return 1 / (np.exp(-A) + 1)

def activation(A):
    return np.log(1 + np.exp(A))

def output(X, W2):
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones,X))
    A = np.dot(X, W2)
    return softmax(A)

def middle_deltas(X,W,out_deltas):
    [W1, W2] = W
    A = np.dot(X, W1)
    no_bias = np.delete(W2, 0, axis=0)
    deriv_act = sigmoid(A)
    sums = np.dot(out_deltas, no_bias.T)
    return sums * deriv_act

def middle_grad(X,W,out_deltas):
    d2 = middle_deltas(X, W, out_deltas)
    grad = np.dot(X.T, d2)
    return grad

def forward_pass(X,W):  #x is a vector, W is a list with 2 arrays having the weights, returns the outputs of the neural network
    [W1,W2]=W
    act=activation(np.dot(X, W1))
    out=output(act,W2)
    return out

def activations_outputs(X,W): #the same as above, but also returns the activations
    [W1,W2] = W
    act=activation(np.dot(X,W1))
    out=output(act,W2)
    return [act,out]

def gradient(X,T,W):
    [act, out] = activations_outputs(X, W)
    D1 = T-out
    mid_grad = middle_grad(X, W, D1)
    ones = np.ones((act.shape[0],1))
    act = np.hstack((ones, act))  # for the bias
    out_grad = np.dot(act.T, D1)
    return [mid_grad, out_grad]

def cost(X,T,W,l):   #the cost function
    Y = forward_pass(X, W)
    [W1, W2] = W
    return np.sum(T * np.log(Y + np.finfo(float).eps)) - l / 2 * (np.sum(np.linalg.norm(W1, axis=0) ** 2) + np.sum(np.linalg.norm(W2, axis=0) ** 2))

def computed_gradient(X,T,W,l):
    [W1, W2] = W
    W1[0,0] = W1[0,0] + 0.000001
    add = cost(X,T,[W1,W2],l)
    W1[0, 0] = W1[0, 0] - 0.000002
    minus = cost(X, T, [W1, W2], l)
    grad = (add - minus)/0.000002
    return grad

def train(X,T,init,l,etta,iterations): #X array (N,D+1), T array (N,K), init is the initial guess for the weights, l regularization parameter
    (N,D) = X.shape
    E_old=-np.inf
    E_new=cost(X,T,init,l)
    W=init
    [W1, W2] = W
    i = 0
    time_elapsed = 0
    while(np.abs(E_new-E_old)>0.001):
        start = time.time()
        print('Starting iteration {}'.format(i))
        E_old=E_new
        (grad1, grad2) = gradient(X, T, W)
        #t = computed_gradient(X,T,W,l)
        grad1=grad1-l*W1
        grad2=grad2-l*W2
        W1=W1+etta*grad1
        W2=W2+etta*grad2
        W=[W1,W2]
        E_new=cost(X,T,W,l)
        print(E_new)
        time_elapsed += (time.time() - start)
        print('Elapsed: {:.3f}s, Remaining: {:.3f}s'.format(time_elapsed,(time_elapsed / (i + 1)) * (iterations - i + 1)))
        i += 1
        if(i > iterations):
            break
    return W

def predict(X,W):   #X must have 1s as first column
    (N,D)=X.shape
    pred=np.zeros(N)
    Y = forward_pass(X, W)
    pred = np.argmax(Y, axis=1)
    return pred

def fuzzy_target(traintarget):
    ''' Transform the indicators from 0 and 1 to values close to 0 and 1'''
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

    def init_weights():
        M = 50
        np.random.seed(1)
        W1 = np.random.randn(traindata.shape[1], M)
        W2 = np.random.randn(M+1, traintarget.shape[1])
        W = [W1, W2]
        return W

    W = train(traindata/255, traintarget, init_weights(), 0.2, 0.00001,1000)
    Y = predict(testdata/255, W)
    res = Y - np.argmax(testtarget, axis=1)
    zeros = 0
    for row in range(res.shape[0]):
        if(res[row] == 0):
            zeros = zeros +1
    print(zeros)

if __name__ == '__main__':
    main()