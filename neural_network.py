# version 1.1

from typing import List
import numpy as np

from operations import *

class NeuralNetwork():
    '''
    A class for a fully connected feedforward neural network (multilayer perceptron).
    :attr n_layers: Number of layers in the network
    :attr activations: A list of Activation objects corresponding to each layer's activation function
    :attr loss: A Loss object corresponding to the loss function used to train the network
    :attr learning_rate: The learning rate
    :attr W: A list of weight matrices. The first row corresponds to the biases.
    '''

    def __init__(self, n_features: int, layer_sizes: List[int], activations: List[Activation], loss: Loss,
                 learning_rate: float=0.01, W_init: List[np.ndarray]=None):
        '''
        Initializes a NeuralNetwork object
        :param n_features: Number of features in each training examples
        :param layer_sizes: A list indicating the number of neurons in each layer
        :param activations: A list of Activation objects corresponding to each layer's activation function
        :param loss: A Loss object corresponding to the loss function used to train the network
        :param learning_rate: The learning rate
        :param W_init: If not None, the network will be initialized with this list of weight matrices
        '''

        sizes = [n_features] + layer_sizes
        if W_init:
            assert all([W_init[i].shape == (sizes[i] + 1, sizes[i+1]) for i in range(len(layer_sizes))]), \
                "Specified sizes for layers do not match sizes of layers in W_init"
        assert len(activations) == len(layer_sizes), \
            "Number of sizes for layers provided does not equal the number of activations provided"

        self.n_layers = len(layer_sizes)
        self.activations = activations
        self.loss = loss
        self.learning_rate = learning_rate
        self.W = []
        for i in range(self.n_layers):
            if W_init:
                self.W.append(W_init[i])
            else:
                rand_weights = np.random.randn(sizes[i], sizes[i+1]) / np.sqrt(sizes[i])
                biases = np.zeros((1, sizes[i+1]))
                self.W.append(np.concatenate([biases, rand_weights], axis=0))

    def forward_pass(self, X) -> (List[np.ndarray], List[np.ndarray]):
        '''
        Executes the forward pass of the network on a dataset of n examples with f features. Inputs are fed into the
        first layer. Each layer computes Z_i = g(A_i) = g(Z_{i-1}W[i]).
        :param X: The training set, with size (n, f)
        :return A_vals: a list of a-values for each example in the dataset. There are n_layers items in the list and
                        each item is an array of size (n, layer_sizes[i])
                Z_vals: a list of z-values for each example in the dataset. There are n_layers items in the list and
                        each item is an array of size (n, layer_sizes[i])
        '''
        
        layers_a_vals = []
        layers_z_vals = []
        for i in range(self.n_layers): # For each layer
            layer_a_vals = []
            layer_z_vals = []
            for j in range(len(X)): # For each input in x
                a_values = []
                z_values = []
                for output in range(np.shape(self.W[i])[1]): 
                    a_val = 0
                    for input in range(np.shape(self.W[i])[0]): 
                        if (input == 0):
                            a_val += self.W[i][input][output]
                        elif (i == 0):
                            a_val += X[j][input - 1] * self.W[i][input][output]
                        else:
                            a_val += layers_z_vals[i - 1][j][input - 1] * self.W[i][input][output]
                    a_values.append(a_val)
                z_values = self.activations[i].value(a_values)
                layer_a_vals.append(a_values)
                layer_z_vals.append(z_values)
            layers_a_vals.append(layer_a_vals)
            layers_z_vals.append(layer_z_vals)
        return layers_a_vals, layers_z_vals

    def backward_pass(self, A_vals, dLdyhat) -> List[np.ndarray]:
        '''
        Executes the backward pass of the network on a dataset of n examples with f features. The delta values are
        computed from the end of the network to the front.
        :param A_vals: a list of a-values for each example in the dataset. There are n_layers items in the list and
                       each item is an array of size (n, layer_sizes[i])
        :param dLdyhat: The derivative of the loss with respect to the predictions (y_hat), with shape (n, layer_sizes[-1])
        :return deltas: A list of delta values for each layer. There are n_layers items in the list and
                        each item is an array of size (n, layer_sizes[i])
        '''

        layers_delta_vals = []
        for i in range(self.n_layers): # For each layer (reversed)
            layer_delta_vals = []
            for j in range(len(A_vals[i])): # For each input in x
                delta_values = []
                actualLayer = self.n_layers - 1 - i
                if (actualLayer == self.n_layers - 1): # IF on the last layer
                    for output in range(np.shape(self.W[actualLayer])[1]):
                        delta_val = dLdyhat[j][output] * self.activations[actualLayer].derivative(A_vals[actualLayer][j][output])
                        delta_values.append(delta_val)
                else: # if i + 1 exists 
                    for input in range(1, np.shape(self.W[actualLayer + 1])[0]): # ignore bias layer
                        weightDeltaSum = 0
                        for output in range(np.shape(self.W[actualLayer + 1])[1]): # access previous layer (next actual layer) with 0
                            weightDeltaSum += layers_delta_vals[0][j][output] * self.W[actualLayer + 1][input][output] 
                        delta_val = weightDeltaSum * self.activations[actualLayer].derivative(A_vals[actualLayer][j][input - 1])
                        delta_values.append(delta_val)
                layer_delta_vals.append(delta_values)
            layers_delta_vals.insert(0, layer_delta_vals)

        return layers_delta_vals

    def update_weights(self, X, Z_vals, deltas) -> List[np.ndarray]:
        '''
        Having computed the delta values from the backward pass, update each weight with the sum over the training
        examples of the gradient of the loss with respect to the weight.
        :param X: The training set, with size (n, f)
        :param Z_vals: a list of z-values for each example in the dataset. There are n_layers items in the list and
                       each item is an array of size (n, layer_sizes[i])
        :param deltas: A list of delta values for each layer. There are n_layers items in the list and
                       each item is an array of size (n, layer_sizes[i])
        :return W: The newly updated weights (i.e. self.W)
        '''

        for i in range(self.n_layers):
            for input in range(np.shape(self.W[i])[0]):
                for output in range(np.shape(self.W[i])[1]):
                    change_weight = 0 
                    for j in range(len(X)):
                        if (i == 0): # input layer
                            if (input == 0):
                                change_weight += deltas[i][j][output] * 1
                            else:
                                change_weight += deltas[i][j][output] * X[j][input - 1]
                        else:
                            if (input == 0):
                                change_weight += deltas[i][j][output] * 1
                            else:
                                change_weight += deltas[i][j][output] * Z_vals[i - 1][j][input - 1]
                    self.W[i][input][output] = self.W[i][input][output] - self.learning_rate * change_weight
        return self.W

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int) -> (List[np.ndarray], List[float]):
        '''
        Trains the neural network model on a labelled dataset.
        :param X: The training set, with size (n, f)
        :param y: The targets for each example, with size (n, 1)
        :param epochs: The number of epochs to train the model
        :return W: The trained weights
                epoch_losses: A list of the training losses in each epoch
        '''

        epoch_losses = []
        for epoch in range(epochs):
            A_vals, Z_vals = self.forward_pass(X)   # Execute forward pass
            y_hat = Z_vals[-1]                      # Get predictions
            L = self.loss.value(y_hat, y)           # Compute the loss
            print("Epoch {}/{}: Loss={}".format(epoch, epochs, L))
            epoch_losses.append(L)                  # Keep track of the loss for each epoch

            dLdyhat = self.loss.derivative(y_hat, y)         # Calculate derivative of the loss with respect to output
            deltas = self.backward_pass(A_vals, dLdyhat)     # Execute the backward pass to compute the deltas
            self.W = self.update_weights(X, Z_vals, deltas)  # Calculate the gradients and update the weights

        return self.W, epoch_losses

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric) -> float:
        '''
        Evaluates the model on a labelled dataset
        :param X: The examples to evaluate, with size (n, f)
        :param y: The targets for each example, with size (n, 1)
        :param metric: A function corresponding to the performance metric of choice (e.g. accuracy)
        :return: The value of the performance metric on this dataset
        '''

        A_vals, Z_vals = self.forward_pass(X)       # Make predictions for these examples
        y_hat = Z_vals[-1]
        metric_value = metric(y_hat, y)     # Compute the value of the performance metric for the predictions
        return metric_value

