
# BMI 203 Final Project


NN.py - neural net with flexible number of layers, neurons. Takes architecture, learning rate, batch size, loss as input
	* feedforward - run input forward to generate output
	* backpropagation - use difference between expected and truth to update weights
	* predict - feedforward only
	* training iteration - split training into batches, update weights by each batch, return average loss
	* score prediction - use loss function to evaluate prediction against truth
	* clear weights - reset all weights without reinitializing neural net
	* also includes utility functions for activation and backprop

main.py - run everything 
	* autoencoder - creates 8x3x8 autoencoder and tests reconstruction of identity matrix
	* test_hyperparms - runs and evaluates model for set of various hyperparameters
	* run_best_model - create ROC plot, train and run on test set using best hyperparams
