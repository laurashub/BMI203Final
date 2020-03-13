import numpy as np


from scripts import io
from scripts import NN

# 1 bit for base identity, 1 bit for purine/pyrimidine
#adenine and guanine are purines, C and T are pyrimidines
rep_dir = {'C':[0, 5], 'A':[1, 4], 'T':[2, 5], 'G':[3, 4]}

def generate_neg_examples(pos, neg):
	pos_len = len(pos[0])
	fixed_neg = set()
	for neg_ex in neg:
		#split into chunks of pos_len
		for i in range(len(neg_ex) - (pos_len - 1)):
			ex = neg_ex[i:i+pos_len]
			#only add elements that are not actual binding sites
			if ex not in pos:
				fixed_neg.add(ex)
	return list(fixed_neg)

def training(nn, input, truth, n_epochs, verbose = False):
	if n_epochs < 10:
		step = 1
	else:
		step = n_epochs / 10
	for epoch in range(n_epochs):
		loss = nn.training_iteration(input, truth)
		if epoch % step == 0 and verbose:
			print(f'Epoch: {epoch}, loss: {loss}')
			#print(nn)
	return loss

def cross_validation(nn, input, truths, n_epochs, n):
	#concat input and truths
	all_data = np.concatenate((input, truths), axis = 1)
	splits = np.split(all_data, n)
	for i in range(n):
		training = np.concatenate([split for j, split in enumerate(splits) if j is not i])
		validate = splits[i]

		loss = training(nn, training[:,:all_data.shape[1]], training[:, all_data.shape[1]:])
		validate = nn.score_prediction(validation, truth)
		print(f'CV {i}, loss: {loss}, validation: {validate}')


def generate_rep(seq):
	"""
	"""
	seq_rep = np.array([])

	for base in seq:
		base_rep = [1 if i in rep_dir[base] else 0 for i in range(6)]
		seq_rep = np.concatenate((seq_rep, base_rep))
	return np.atleast_2d(seq_rep)

def classify_sites():
	pos = io.read_seqs('data/rap1-lieb-positives.txt')
	neg = io.read_seqs('data/yeast-upstream-1k-negative.fa')

	neg_examples = generate_neg_examples(pos, neg)

	pos_test = pos[:5]
	neg_test = neg_examples[:5]

	all_examples = pos_test + neg_test
	test_set = np.concatenate([generate_rep(ex) for ex in all_examples])
	test_truth = np.atleast_2d(np.concatenate((np.ones(5), np.zeros(5)))).T

	print(test_set.shape)
	print(test_truth.shape)

	my_NN = NN.NeuralNetwork(architecture = [102, 50, 25, 5, 1], lr = 0.1, batch_size = 2)
	#print(my_NN)
	#print(my_NN)
	#print(my_NN.predict(test_set[:2, :].T))
	training(my_NN, test_set[:2, :], test_truth[:2,:], 100, verbose = True)


#print(neg)

#neg_examples = generate_neg_examples(pos, neg)
#assert all([x not in pos for x in neg_examples])

def autoencoder():
	"""
	Can my 8 x 3 x 8 autoencoder learn to recreate an 8 x 8 matrix
	"""
	my_NN = NN.NeuralNetwork(lr = 0.1, batch_size = 1)
	print(my_NN)

	id_mat = np.identity(8)

	training(my_NN, id_mat, id_mat, 100000, verbose = True)

	test = id_mat #np.random.rand(8,8)
	print(test)
	predicted = my_NN.predict(test)
	print(predicted)
	#cross_validation(my_NN, input, input, 1, 10)
	print(test - predicted)


	#print(my_NN)
#classify_sites()
autoencoder()



