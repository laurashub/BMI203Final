import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from scripts import io
from scripts import NN

# 1 bit for base identity, 1 bit for purine/pyrimidine
rep_dir = {'C':[0, 5], 'A':[1, 4], 'T':[2, 5], 'G':[3, 4]}
rc_dir = {'C':'G', 'G':'C', 'A':'T', 'T':'A'}


def generate_neg_examples(pos, neg):
	"""
	Read negative examples, split into same size as positive
	"""
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

def training(nn, input, truth, n_epochs, verbose = False, shuffle = True, balance_classes = False, class_members = 2000):
	"""
	Train nn on input for n_epochs epochs, returns the loss at the end
	"""
	if n_epochs < 10:
		step = 1
	else:
		step = n_epochs / 10
	prev_loss = float("inf")
	for epoch in range(n_epochs):
		#single weight update
		loss = nn.training_iteration(input, truth, shuffle = shuffle, balance_classes = balance_classes, 
			class_members = class_members)
		#potential to end early
		if loss < 0.001:
			return loss
		prev_loss = loss
		if epoch % step == 0 and verbose:
			print(f'Epoch: {epoch}, loss: {loss}')
	return loss

def cross_validation(nn, input, truths, n_epochs, n, batch_size, class_members = 2000, 
	balance_classes = False, plot = False, title = ""):
	"""
	Test hyperparameters by performing cross validation, potentially plot ROC
	"""

	#concat input and truths
	all_data = np.concatenate((input, truths), axis = 1)

	#split into n sets, keep class balanced
	if balance_classes:
		classes = {}
		classifications = set([x[0] for x in all_data[:,input.shape[1]:]])
		for c in classifications:
			indices = [x[0] == c for x in all_data[:,input.shape[1]:]]
			classes[c] = all_data[indices, :]
			classes[c] = np.array_split(classes[c], n)
		splits = []
		for i in range(n):
			splits.append(np.concatenate([classes[c][i] for c in classifications]))

	#split with no regard for classes
	else:
		splits = np.array_split(all_data, n)

	#hold out each set and train
	valid_predictions = {}

	total_validation = 0
	for i in range(n):
		training_set = np.concatenate([split for j, split in enumerate(splits) if j is not i])
		validation = splits[i]

		#reinitialize to starting weights
		nn.clear_weights()

		#get loss
		loss = training(nn, training_set[:,:input.shape[1]], training_set[:, input.shape[1]:],
			n_epochs, balance_classes = balance_classes, class_members = class_members)


		val_prediction = nn.predict(validation[:,:input.shape[1]])

		validate = nn.score_prediction(val_prediction, validation[:, input.shape[1]:])
		fpr, tpr, _ = roc_curve(validation[:, input.shape[1]:], val_prediction, pos_label = 1)
		valid_predictions[i] = (fpr, tpr)

		total_validation += validate
		print(f'CV {i}, loss: {loss}, validation: {validate}')

	if plot:

		#run once on whole set
		nn.clear_weights()

		loss = training(nn, input, truths,
			n_epochs, balance_classes = balance_classes, class_members = class_members)
		
		val_prediction = nn.predict(input)
		fpr, tpr, _ = roc_curve(truths, val_prediction, pos_label = 1)
		
		valid_predictions['all'] = (fpr, tpr)


		plt.figure()
		for i in valid_predictions:
			fpr, tpr = valid_predictions[i]
			plt.plot(fpr, tpr, label=f'ROC curve {i}: {auc(fpr, tpr)}' % auc(fpr, tpr))
		plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title(title)
		plt.legend(loc="lower right")
		plt.savefig(title + '.png', dpi = 200)
		#plt.show()

	return total_validation / n


def generate_rep(seq):
	"""
	Bit representaiton of sequence
	"""
	seq_rep = np.array([])

	for base in seq:
		base_rep = [1 if i in rep_dir[base] else 0 for i in range(6)]
		seq_rep = np.concatenate((seq_rep, base_rep))
	return np.atleast_2d(seq_rep)

def classify_sites(lr, n_epochs, architecture, batch_size, class_members, loss_dict = None, 
	predict_test = False, plot = False):

	"""
	Run full classification w cross validation
	"""

	np.random.seed(666)
	layers = "_".join([str(layer) for layer in architecture])
	trial_name = f'lr-{lr}_epochs-{n_epochs}_arch-{layers}_bs-{batch_size}_cm-{class_members}'

	pos = io.read_seqs('data/rap1-lieb-positives.txt')
	neg = io.read_seqs('data/yeast-upstream-1k-negative.fa')

	neg = generate_neg_examples(pos, neg)

	#balance classes - downsample negative and duplicate positive examples until even
	# this is naive, might update

	#downsample negative, too many to run in any realistic amt of time
	#currently select 5000, then sample both classus until 2000 so I dont generate all 100k+ representations -> change?
	neg = [neg[i] for i in np.random.choice(len(neg), 5000, replace = False)]

	#generate 1-hot encoding + purine/pyrimidine classification
	pos_examples = np.concatenate([generate_rep(x) for x in pos])
	neg_examples = np.concatenate([generate_rep(x) for x in neg])

	#create full training + truth, 1 if binding sequence else 0
	training_set = np.concatenate((pos_examples, neg_examples))
	training_truth = np.atleast_2d(np.concatenate((np.ones(len(pos_examples)), np.zeros(len(neg_examples))))).T

	#train
	#using supplied parameters
	my_NN = NN.NeuralNetwork(architecture = architecture, lr = lr, batch_size = batch_size)

	if predict_test:
		#retrain net on full dataset
		my_NN = NN.NeuralNetwork(architecture = architecture, lr = lr, batch_size = batch_size)
		#nn, input, truth, n_epochs, verbose = False, shuffle = True, balance_classes = False, class_members = 2000)
		training(my_NN, training_set, training_truth, n_epochs, verbose = True, balance_classes = True, class_members = 2000)

		rank_tests(my_NN)

	else:
		valid_loss = cross_validation(my_NN, training_set, training_truth, n_epochs, 
			5, batch_size, class_members = class_members, balance_classes = True, plot = plot, title = trial_name)

		if loss_dict is None:
			#Train and print out a few example classifications

			my_NN = NN.NeuralNetwork(architecture = architecture, lr = lr, batch_size = batch_size)
			#nn, input, truth, n_epochs, verbose = False, shuffle = True, balance_classes = False, class_members = 2000)
			training(my_NN, training_set, training_truth, n_epochs, verbose = True, balance_classes = True, class_members = 2000)
		
			#print out a few example to check what it learned
			test = np.random.choice(pos_examples.shape[0], 5)
			pos_test = pos_examples[test, :]
			neg_test = neg_examples[test, :]
			test_truth = np.atleast_2d(np.concatenate((np.ones(5), np.zeros(5)))).T

			predictions = my_NN.predict(np.concatenate((pos_test, neg_test)))
			
			for x, y in zip(test_truth, predictions):
				print(x, y)

		else:
			#update loss_dict
			loss_dict[trial_name] = valid_loss
		print(trial_name, valid_loss)
	

def rank_tests(nn):
	"""
	Use nn to classify all sequences in test set
	"""
	test_seqs = io.read_seqs('data/rap1-lieb-test.txt')

	test_set = np.concatenate([generate_rep(x) for x in test_seqs])
	predicted = nn.predict(test_set)

	with open('shub_prediction.tsv', 'wt') as f:
		tsv_writer = csv.writer(f, delimiter='\t')
		for seq, prediction in zip(test_seqs, predicted):
			tsv_writer.writerow([seq, prediction.item(0)])
			print(seq, prediction.item(0))


def autoencoder(plot = True):
	"""
	Can my 8 x 3 x 8 autoencoder learn to recreate 8 x 8 identity matrix
	"""
	my_NN = NN.NeuralNetwork(lr = 0.1, batch_size = 1)

	id_mat = np.identity(8)
	training(my_NN, id_mat, id_mat, 100000, verbose = True)

	predicted = my_NN.predict(id_mat)
	print(predicted)

	if plot:
		fig, (ax1, ax2) = plt.subplots(1,2, figsize=(22,10))
		sns.heatmap(id_mat, ax=ax1)
		sns.heatmap(predicted, ax=ax2)
		ax1.set_title("Input")
		ax2.set_title("Reconstructed")
		plt.savefig('id_matrix_reconstruction.png', dpi=200)

def test_hyperparams():
	"""
	Train multiple models with differect hyperparameters
	"""
	loss_dict = {}

	#naive combinatorial search
	lrs = [0.1, 0.05, 0.01]
	#n_epochs = [1000, 5000, 10000]
	architectures = [[102, 20, 1], [102, 50, 1], [102, 50, 20, 1]]
	sample_sizes = [1000, 2000, 3000]
	#batch_sizes = [10, 20, 50]
	bs = 10

	#generate 27 trained models
	for lr in lrs:
		for arch in architectures:
			for sample_size in sample_sizes:
				classify_sites(lr, 1000, arch, 10, sample_size, loss_dict, plot = False)

	with open('loss_dict.txt', 'w+') as f:
		f.write(str(loss_dict))

def run_best_model():
	"""
	Using best model from loss_dict, run CV to generate ROC plot, train on all data, classify test seqs
	"""
	loss_dict = eval(open('loss_dict.txt', 'r').read())
	print(loss_dict)

	#get hyperparams with lowest loss
	best_hp, best_loss = None, float("inf")
	for hyperparams in loss_dict:
		if loss_dict[hyperparams] < best_loss:
			best_loss = loss_dict[hyperparams]
			best_hp = hyperparams

	#parse hyperparam title 
	tokens = [x.split('-')[1] if len(x.split('-')) > 1 else x.split('-')[0] for x in best_hp.split('_')]
	print(tokens)

	#run CV to generate graph
	lr = float(tokens[0])
	n_epochs = int(tokens[1])
	architecture = [int(x) for x in tokens[2:-2]]
	batch_size = int(tokens[-2])
	class_members = int(tokens[-1])

	classify_sites(lr, n_epochs, architecture, batch_size, class_members, loss_dict = None, 
	predict_test = False, plot = True)

	#generate values for test set
	classify_sites(lr, n_epochs, architecture, batch_size, class_members, loss_dict = None, predict_test = True)

	#print(best_hp, best_loss)

#make results consistent
np.random.seed(27)
#autoencoder()

#test_hyperparams()
run_best_model()

