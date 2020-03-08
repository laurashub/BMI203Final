import numpy as np

from scripts import io
from scripts import NN

#print(len(io.read_seqs('data/rap1-lieb-positives.txt')))
print(np.ones((8,8)))
my_NN = NN.NeuralNetwork()

print(my_NN)

input = np.random.rand(10,8)
print(input.shape)
#assert False
for i in range(10000):
	my_NN.training_iteration(input, input, i)
final = my_NN.predict(input)
print(f"Input value: {input}, \nreconstructed: {final}")
