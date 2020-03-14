from scripts import NN
from scripts.__main__ import *


# Note: not really using, see __main__.py and write-up

def test_encoder():
	"""
	Can my 8 x 3 x 8 autoencoder learn to recreate 8 x 8 identity matrix
	"""
	my_NN = NN.NeuralNetwork(lr = 0.1, batch_size = 1)

	id_mat = np.identity(8)
	training(my_NN, id_mat, id_mat, 1000, verbose = True)

	predicted = my_NN.predict(id_mat)

	assert (it_mat - predicted).mean() < 0.005