import numpy as np

def forward(X, W1, W2):
    '''
        Inputs:
        NxD matrix X of features, where N is the number of samples and D is the number of feature dimensions,
        MxD matrix W1 of weights between the first and second layer of the network, where M is the number of hidden neurons, and
        1xM matrix W2 of weights between the second and third layer of the network, where there is a single neuron at the output layer

        Outputs:
        Nx1 vector y_pred containing the outputs at the last layer for all N samples, and
        NxM matrix Z containing the activations for all M hidden neurons of all N samples.
    '''
    a0 = np.dot(W1, X.T)      # hidden layer
    z0 = np.tanh(a0.T)        # activation
    a1 = np.dot(z0, W2.T)     # output layer

    return z0, a1