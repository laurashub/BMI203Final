import numpy as np


class NeuralNetwork():

    def __init__(self, architecture = [8, 3, 8], activation_funcs = None, 
        lr = .05, seed = 1,  wd = .00001, batch_size = 20, loss = 'MSE'):

        #set seed
        self.seed = seed
        np.random.seed(self.seed)

        #set hyperparams
        self.arch = architecture
        self.lr = lr
        self.wd = wd
        self.bs = batch_size
        self.lf = loss_funcs[loss]
        self.dloss = deriv[self.lf]

        self.params = {}

        #if no activations provided, default to sigmoid
        if activation_funcs is None:
            activation_funcs = []
            for i in range(len(self.arch) - 1):
                activation_funcs.append(sigmoid)


        #initialize layers and info (weights, activation func, gradient)
        for i, layer in enumerate(self.arch[1:]):
            self.params[i] = {}

            self.params[i]['weights'] = np.random.rand(self.arch[i], layer)
            self.params[i]['activation'] = activation_funcs[i]
            self.params[i]['gradient'] = deriv[activation_funcs[i]]


    def feedforward(self, input):
        '''
        Feed forward propagation through network
        '''
        activation = input
        for layer in self.params:
            z = np.dot(activation, self.params[layer]['weights'])
            activation = self.params[layer]['activation'](z)
            self.params[layer]['a'] = activation #save activation, move onto next level
        return activation


    def backpropagation(self, output, truth, input):
        """
        Update weights based on current error
        """

        output_layer = self.params[max(self.params)]

        #partial deriv of error wrt output weights
        error = deriv[self.lf](output, truth)
        output_layer['weight_update'] = error * output_layer['gradient'](output) 

        #backpropagation for gradient descent
        for layer in range(max(self.params) - 1, -1, -1):
            current_layer = self.params[layer]
            right_layer = self.params[layer + 1]

            error = right_layer['weight_update'].dot(right_layer['weights'].T)
            current_layer['weight_update'] = error * current_layer['gradient'](current_layer['a'])
 
        #update weights based on calculated weight update
        prev_activate = input
        for layer in self.params:
            current_layer = self.params[layer]

            current_layer['weights'] -= self.lr * prev_activate.T.dot(current_layer['weight_update'])
            prev_activate = current_layer['a']


    def predict(self, input):
        """
        #Feed forward, no backprop
        """
        return self.feedforward(input)


    def training_iteration(self, input, truth, shuffle = True, balance_classes = False, class_members = 2000 ):
        #concat truth to input to keep them together during split/shuffle
        full_input = np.concatenate((input, truth), axis = 1 )

        #split by class
        if balance_classes:
            classes = set([x[0] for x in full_input[:,input.shape[1]:]])

            class_inputs = []
            for c in classes:
                indices = [x[0] == c for x in full_input[:,input.shape[1]:]]
                class_input = full_input[indices, :]
                class_inputs.append(class_input[np.random.choice(class_input.shape[0],
                    class_members, replace = True),:])
            full_input = np.concatenate(class_inputs)

        #shuffle input  
        if shuffle:
            np.random.shuffle(full_input)

        #split into batches
        #is this is larger than the batch size, split up
        if self.bs < full_input.shape[0]:
            batches = np.split(full_input, int(full_input.shape[0]/self.bs))
        else:
            batches = [full_input]

        #update weights by batch, average loss across all batches
        total_loss = 0
        for i, batch in enumerate(batches):
            batch_input = batch[:,:input.shape[1]]
            batch_truth = batch[:,input.shape[1]:]
            output = self.feedforward(batch_input)
            total_loss += self.lf(output, batch_truth)
            self.backpropagation(output, batch_truth, batch_input)
        return total_loss/len(batches)

    def score_prediction(self, predict, truth):
        return self.lf(predict, truth)

    def clear_weights(self):
        #make sure theyre reinitialized to the same
        np.random.seed(self.seed)
        for layer in self.params:
            current_layer = self.params[layer]
            x, y = current_layer['weights'].shape[0], current_layer['weights'].shape[1]
            current_layer['weights'] = np.random.rand(x, y)

    def __repr__(self):
        str_rep = ""
        for layer, info in self.params.items():
            str_rep += f"Layer {layer+1}: \n \
            {info['weights'].shape[0]} inputs \n \
            {info['weights'].shape[1]} outputs\n \
            {info['weights']}\n"
        return str_rep

#define activation funcs to be used
def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a

def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(yhat, y):
    return (((y - yhat)**2).mean(axis = 1) / 2).mean()

def mse_gradient(yhat, y):
    return yhat - y

deriv = {sigmoid : sigmoid_gradient, mse: mse_gradient}
loss_funcs = {'MSE':mse}

  
