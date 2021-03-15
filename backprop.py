from forward import forward
import numpy as np

def backprop(X, y, M, iters, eta):
    '''
        Performs training using backpropagation (and calls the forward function as it iterates). 
        Construct the network in this function --> create weight matrices + initialize to small random numbers
        iterate: pick a training sample, compute the error at the output, backpropagate to the hidden layer
        update the weights with the resulting error.

        Inputs:
        NxD matrix X of features, N = number of samples and D = number of feature dimensions, an Nx1 vector y containing the ground-truth labels for the N samples,
        a scalar M containing the number of hidden neurons to use, a scalar iters defining how many iterations to run (one sample used in each), and a scalar eta defining the learning rate to use.
        
        Outputs:
        W1 and W2, defined as above for forward, and
        itersx1 vector error_over_time that contains the error on the sample used in each iteration.
    '''

    # initiaize wieghts
    W1 = np.random.randn(M, 11) * 0.01
    W2 = np.random.randn(1, M)  * 0.01 
    error_over_time = []

    # train network
    for iter in range(iters): 

        i = np.random.randint(0,800)                    # take random sample 
        Xi = X[i,0:11].reshape(1,11)                    # extract feature dimensions
        
        Z, y_pred = forward(Xi, W1, W2)                 # forward propagation
        
        o_error = y_pred - y[i, 0]                      # calcuate error of o/p units
        W2 -= eta * Z * o_error                         # update wieghts onto o/p layer

        h_error = (1 - np.tanh(Z)**2) * o_error * W2    # calcuate error of hidden units 
        W1 -= eta * np.dot(Xi.T, h_error).T             # update wieghts of hidden units

        error_over_time.append(abs(np.asscalar(o_error)))
    
    return W1, W2, error_over_time