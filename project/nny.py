import numpy as np
import mnist
import time

def indicators(t):     # returns array with indicator vectors as rows
    (N,)=t.shape
    if np.min(t)>0:
        categories=np.max(t)
        ind=np.zeros((N,categories))
        for n in range(0,N):
            ind[n,t[n]-1]=1
    else:
        categories=np.max(t)+1
        ind=np.zeros((N,categories))
        for n in range(0,N):
            ind[n,t[n]]=1
    return ind

def softmax(A):
    (N,)=A.shape
    max=np.max(A)
    max_array=max*np.ones(N)
    modA=A-max_array
    expA=np.exp(modA)
    denom=np.sum(expA)
    sm=expA/denom
    return sm

def sigmoid(A):
    (N,)=A.shape
    sig=np.exp(-A)
    sig=sig+np.ones(N)
    sig=1./sig
    return sig

def activations(x,w1):   #x must have 1 as first element, w1 is the (M,D+1) array with the first layer's weights, including bias
    A=np.dot(w1,x)
    (N,)=A.shape
    I=np.ones(N)
    value=np.exp(A)+I
    return np.log(value)

def outputs(x,w2):  #x is the output of the activations function, w2 (K,M+1) array with second layer weights with bias
    x=np.hstack((1,x))  # one is for the bias
    A=np.dot(w2,x)
    return softmax(A)

def middle_deltas(x,w,out_deltas): #x is a training datapoint, w is a list with 2 arrays having the weights, out_deltas are the output level deltas
    [w1,w2]=w
    A=np.dot(w1,x)   # x must have 1's as first element
    no_bias=np.delete(w2, 0, axis=1)  #remove bias parameters
    deriv_act=sigmoid(A)
    sums=np.dot(no_bias.T,out_deltas)
    return sums*deriv_act

def middle_grad(x,w,out_deltas):# the same as above, returns the first layer gradients
    d2=middle_deltas(x,w,out_deltas)
    (M,)=d2.shape
    (D,)=x.shape
    x.shape=(D,1)  #x must have 1 as first element
    d2.shape=(M,1)
    grad=np.dot(d2,x.T)
    return grad

def forward_pass(x,W):  #x is a vector, W is a list with 2 arrays having the weights, returns the outputs of the neural network
    [w1,w2]=W
    act=activations(x,w1)
    out=outputs(act,w2)
    return out

def activations_outputs(x,W): #the same as above, but also returns the activations
    [w1,w2]=W
    act=activations(x,w1)
    out=outputs(act,w2)
    return [act,out]

def gradient(x,t,W):  #t is an indicator vector, returns a list of 2 arrays with all the gradients
    [act,out]=activations_outputs(x,W)
    d1=t-out
    mid_grad=middle_grad(x,W,d1)
    act=np.hstack((1,act))  #for the bias
    (K,)=d1.shape
    (M,)=act.shape
    d1.shape=(K,1)
    act.shape=(M,1)
    out_grad=np.dot(d1,act.T)
    return [mid_grad,out_grad]

def cost(X,T,W,l):   #the cost function
    (N,D)=X.shape
    [w1,w2]=W
    sum=0
    for n in range(0,N):
        y=np.log(forward_pass(X[n],W))
        y=y*T[n]
        sum=sum+np.sum(y)
    return sum - l*(np.sum(np.linalg.norm(w1, axis=0) ** 2) + np.sum(np.linalg.norm(w2, axis=0) ** 2))/2

def train(X,T,init,l,etta,iterations): #X array (N,D+1), T array (N,K), init is the initial guess for the weights, l regularization parameter
    (N,D)=X.shape
    E_old=-np.inf
    E_new=cost(X,T,init,l)
    w=init
    [w1,w2]=w
    i = 0
    time_elapsed = 0
    while(np.abs(E_new-E_old)>0.001):
        start = time.clock()
        print('Starting iteration {}'.format(i))
        E_old=E_new
        grad1=np.zeros(w1.shape)
        grad2=np.zeros(w2.shape)
        for n in range(0,N):
            [partial_grad1,partial_grad2]=gradient(X[n],T[n],w)
            grad1=grad1+partial_grad1
            grad2=grad2+partial_grad2
        grad1=grad1-l*w1
        grad2=grad2-l*w2
        w1=w1+etta*grad1
        w2=w2+etta*grad2
        w=[w1,w2]
        E_new=cost(X,T,w,l)
        print('Cost: {}'.format(E_new))
        time_elapsed += (time.clock() - start)
        print('Elapsed: {:.3f}s, Remaining: {:.3f}s'.format(time_elapsed, (time_elapsed / (i + 1)) * (iterations - i + 1)))
        i += 1
        if (i > iterations):
            break
    return w

def predict(X,W):   #X must have 1s as first column
    (N,D)=X.shape
    pred=np.zeros(N)
    for n in range(0,N):
        y=forward_pass(X[n],W)
        pred[n]=np.argmax(y)
    return pred

def main():
    #traindata, traintarget, testdata, testtarget = cifar10.load_cifar('./cifar-10-batches-py')
    #traindata, traintarget, testdata, testtarget = mnist.load_mnist('./mnist')
    traindata, traintarget, testdata, testtarget = mnist.unpickle_mnist()

    def init_weights():
        M = 50
        np.random.seed(1)
        W1 = np.random.randn(traindata.shape[1], M)
        W2 = np.random.randn(M+1, traintarget.shape[1])
        W1 = W1.T
        W2 = W2.T
        W = [W1, W2]
        return W

    W = train(traindata/255, traintarget, init_weights(), 0.2, 0.0001, 100)
    Y = predict(testdata/255, W)
    res = Y - np.argmax(testtarget, axis=1)
    zeros = 0
    for row in range(res.shape[0]):
        if(res[row] == 0):
            zeros = zeros +1
    print(zeros)

if __name__ == '__main__':
    main()