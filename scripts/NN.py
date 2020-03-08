import numpy as np


class NeuralNetwork:


    def __init__(self, architecture = [8, 3, 8], activation_funcs = None, 
    	lr = .05, seed = 1,  error_rate = 0, bias = 1, wd = .00001):

    	#set seed
    	np.random.seed(seed)

    	#set hyperparams
    	self.arch = architecture
    	self.lr = lr
    	self.bias = bias
    	self.wd = wd

    	self.params = {}

    	if activation_funcs is None:
    		self.activation_funcs = []
    		for l in self.arch[1:]:
    			self.activation_funcs.append(sigmoid)
    	else:
    		self.activation_funcs = activation_funcs

    	#create layers - first is input, not neurons
    	for i, (layer, func) in enumerate(zip(self.arch[1:], self.activation_funcs)):
    		"""
    		Weight matrix dimensions are input_nodes x output_nodes
    		"""
    		self.params[i] = {}
    		self.params[i]['weights'] = np.random.rand(self.arch[i], layer)
    		self.params[i]['activation'] = func
    		self.params[i]['dactivation'] = deriv[func]

    def make_weights(self):
    	return
    	#beware of symmetry 


    def feedforward(self, input):                          
    	caches = []                     
       
    	for layer, info in self.params.items():
    		new_input = input
    		input, z = self.single_forward(new_input, **info)
    		info['z'] = z #save x for backpropagation
    		info['a'] = input

    	return input

    def single_forward(self, input, weights, activation, **kwargs):
    	z = np.matmul(input, weights)
    	output = activation(z)
    	return output, z

    def single_backward(self, layer):
    	current_layer = self.params[layer]
    	right_layer = self.params[layer + 1]

    	dz = current_layer['dactivation'](current_layer['z'])
    	return np.multiply(np.matmul(right_layer['delta'], right_layer['weights'].T), dz)

    def backpropagation(self, input, predicted, actual):

    	#first, calculate the error and the output layer change
    	error = predicted - actual #change

    	output_layer = self.params[max(self.params)]
    	#use error to update output layer
    	dz = output_layer['dactivation'](output_layer['z'])
    	output_layer['delta'] = error * dz

    	#calculate change for all the other layers
    	for layer in range(max(self.params) - 1, -1, -1):
    		self.params[layer]['delta'] = self.single_backward(layer)

    	#update the weights according to calculated delta
    	a = input
    	for layer in self.params:
    		a = np.atleast_2d(a)
    		ad = np.dot(a.T,self.params[layer]['delta'])
    		self.params[layer]['weights'] -= self.lr * ad
    		a = self.params[layer]['a']

    def training_iteration(self, input, truth, iter):
    	output = self.feedforward(input)
    	loss = np.sum(output - truth)
    	print(f"Training {iter}, loss: {loss}")
    	self.backpropagation(input, output, truth)

    def fit(self):
    	return

    def predict(self, input):
    	"""
    	Feed forward, no backprop
    	"""
    	return self.feedforward(input)

    def compute_loss(self, loss_function):
    	return

    def __repr__(self):
    	str_rep = ""
    	for layer, info in self.params.items():
    		str_rep += f"Layer {layer+1}: \n \
    		{info['weights'].shape[0]} inputs \n \
    		{info['weights'].shape[1]} outputs\n \
    		{info['weights']}\n"
    	return str_rep

    def print_vals(self, kwargs):
    	for key,val in kwargs.items():
    		print(key,val)

def activation(x):
	return

#define activation funcs to be used
def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a

def sigmoid_gradient(x):
	return sigmoid(x) * (1 - sigmoid(x))

deriv = {sigmoid : sigmoid_gradient}
    
