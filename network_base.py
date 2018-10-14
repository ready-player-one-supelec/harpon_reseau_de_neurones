import numpy as np

def sigmoid(x):
    """Activation function"""
    return  1/(1+np.exp(-x)) #On choisit cette sigmoide car la dérivée est simple

def dsigmoid(m): #Work in progress do not use
    return m * (np.ones(len(m)) - m)

def tanh(x):
    """Activation function"""
    return 1.7159*np.tanh(2*x/3)

def dtanh(m):
    a = 1.7159
    b = 1/(a**2)
    return a*(2/3)*(np.ones(len(m))-b*m*m)

def front_prop(inputs, network, weights, bias, activation=sigmoid):
    """Compute the output of the network
    Parameters:
        inputs: List of n Arrays input of size m
        network: List of p layers sizes
        weights: List of p Arrays of layer[k]*layer[k+1] weights
        bias: List of p Arrays of layer[k] bias
        activation: Activation function
    Output:
        list_out: List of p+1 Arrays of each layer output (and initial input)
    """
    #les weights et biais sont comptés avec leurs colonnes de sortie
    list_out = [np.zeros(network[k]) for k in range(len(network))]
    list_out = [np.array(inputs)] + list_out
    for col in range(len(network)):
        sig_in = np.dot(list_out[col], weights[col]) + bias[col]
        for per, sig_in_per in enumerate(sig_in):
            list_out[col+1][per] = activation(sig_in_per)
    return list_out

def backprop(inputs, th_outputs, reseau, weights, bias, derivee=dsigmoid, activation=sigmoid):
    """Compute the network bias and weight gradient
    Parameters:
        inputs: List of n Arrays input of size m
        th_output:
        network: List of p layers sizes
        weights: List of p Arrays of layer[k]*layer[k+1] weights
        bias: List of p Arrays of layer[k] bias
        # derivative:
    Output:
        grad_weight:
        grad_bias:

    """
    #backpropagation:dc/daijk=dc/dbjk *Yik-1 et dc/bik=SUMj[dc/dbjk+1*aijk+1] *Yik+1(1-Yik+1)
    #matriciellement DAk=DBk.Yk-1 et DBk=Ak+1 . DBk+1*Yk+1 *(1-Yk+1)
    list_out = front_prop(inputs, reseau, weights, bias, activation)
    grad_weight = [weights[k]*0 for k in range(len(weights))]
    grad_bias = [bias[k]*0 for k in range(len(bias))]
    #init
    grad_bias[-1] = derivee(list_out[-1]) * (list_out[-1]-np.array(th_outputs))
    grad_weight[-1] = np.dot(list_out[-2][None].T, grad_bias[-1][None])
    #recurrence
    for col in range(len(list_out)-3, -1, -1):
        grad_bias[col] = np.dot(weights[col+1], grad_bias[col+1]) * derivee(list_out[col+1])
        grad_weight[col] = np.dot(list_out[col][None].T, grad_bias[col][None])
    return grad_weight, grad_bias, np.linalg.norm(th_outputs-list_out[-1])/2

def random_w_b(inputs, network):
    """Create random weights and bias for the network
    Parameters:
        inputs: List of n Arrays input of size m
        network: List of p layers sizes
    Output:
        weights: List of p Arrays of layer[k]*layer[k+1] weights
        bias: List of p Arrays of layer[k] bias
    """
    weights = [2*np.random.random((len(inputs), network[0]))-np.ones((len(inputs), network[0]))]+[2*np.random.random((network[k], network[k+1]))-np.ones((network[k], network[k+1])) for k in range(len(network)-1)]
    # weights = [1 / np.sqrt(len(inputs)) * np.random.randn(len(inputs), network[0])] + [1 / np.sqrt(len(inputs)) * np.random.randn(network[k], network[k+1]) for k in range(len(network)-1)]
    # bias = [np.zeros(network[k]) for k in range(len(network))]
    bias = [1 / np.sqrt(len(inputs)) * np.random.randn(network[k]) for k in range(len(network))]
    return weights, bias

def save_network(network, weights, bias, filename):
    """Save network parameters in a file
    Parameters:
        network: List of p layers sizes
        weights: List of p Arrays of layer[k]*layer[k+1] weights
        bias: List of p Arrays of layer[k] bias
        filename: String naming the saved file
    """
    with open(filename, "w") as file:
        for layer in network:
            file.write(f"{layer} ")
        file.write('\n')
        for layer in weights:
            for line in layer:
                for value in line:
                    file.write(f"{value} ")
                file.write("|")
            file.write('$')
        file.write('\n')
        for layer in bias:
            for value in layer:
                file.write(f'{value} ')
            file.write('|')
    return True

def load_network(filename):
    """Load a network configuration from a file
    Parameters:
        filename: String describing file to load
    Outputs:
        network: List of p layers sizes
        weights: List of p Arrays of layer[k]*layer[k+1] weights
        bias: List of p Arrays of layer[k] bias
    """
    with open(filename, "r") as file:
        # NETWORK
        network = file.readline().split(' ')[:-1]
        network = [int(x) for x in network]
        # WEIGHTS
        weights = file.readline().split('$')[:-1]
        weights = [line.split('|')[:-1] for line in weights]
        weights = [[value.split(' ')[:-1] for value in line] for line in weights]
        weights = [[[float(x) for x in value] for value in line] for line in weights]
        weights = [np.array(layer) for layer in weights]
        # BIAS
        bias = file.readline().split('|')[:-1]
        bias = [layer.split(' ')[:-1] for layer in bias]
        bias = [[float(x) for x in layer] for layer in bias]
        bias = [np.array(layer) for layer in bias]
    return network, weights, bias
