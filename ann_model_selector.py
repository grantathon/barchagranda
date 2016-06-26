import tensorflow as tf
import pandas as pd
import numpy as np
import json
import random
from ArtificialNeuralNetworkClassifier import ArtificialNeuralNetworkClassifier
from sklearn.cross_validation import KFold
from datetime import datetime
from pprint import pprint
from read_numerai import *


def create_parameter_sets(hidden_layer_spaces, dropout_rate_space):
	parameter_sets = []

	layer_neuron_scenarios = []
	dropout_rate_scenarios = []
	dropout_rates = np.linspace(dropout_rate_space[0], dropout_rate_space[1], dropout_rate_space[2])

	# Go through all hidden layer spaces to construct ANN parameters
	for hls in hidden_layer_spaces:
		num_hidden_layers = len(hls)
		num_dropout_rates = len(dropout_rates)

		if(num_hidden_layers == 0):
			parameter_dict = {}
			parameter_dict['hidden_layers'] = None
			parameter_dict['dropout_rates'] = None
			parameter_sets.append(parameter_dict)
		elif(num_hidden_layers == 1):
			for i in range(num_dropout_rates):
				for j in range(hls[num_hidden_layers-1][0], hls[num_hidden_layers-1][1]+1):
					parameter_dict = {}
					parameter_dict['hidden_layers'] = [j]
					parameter_dict['dropout_rates'] = [round(dropout_rates[i], 4)]
					parameter_sets.append(parameter_dict)
		elif(num_hidden_layers == 2):
			for i in range(num_dropout_rates):
				for m in range(num_dropout_rates):
					for j in range(hls[num_hidden_layers-1][0], hls[num_hidden_layers-1][1]+1):
						for k in range(j, hls[num_hidden_layers-2][1]+1):
							parameter_dict = {}
							parameter_dict['hidden_layers'] = [k, j]
							parameter_dict['dropout_rates'] = [round(dropout_rates[i], 4), round(dropout_rates[m], 4)]
							parameter_sets.append(parameter_dict)
		elif(num_hidden_layers == 3):
			for i in range(num_dropout_rates):
				for n in range(num_dropout_rates):
					for p in range(num_dropout_rates):
						for j in range(hls[num_hidden_layers-1][0], hls[num_hidden_layers-1][1]+1):
							for k in range(j, hls[num_hidden_layers-2][1]+1):
								for m in range(k, hls[num_hidden_layers-3][1]+1):
									parameter_dict = {}
									parameter_dict['hidden_layers'] = [m, k, j]
									parameter_dict['dropout_rates'] = [round(dropout_rates[i], 4), round(dropout_rates[n], 4), \
										round(dropout_rates[p], 4)]
									parameter_sets.append(parameter_dict)
		else:
			raise NotImplementedError('Does not support %i hidden layers' % num_hidden_layers)

	return parameter_sets


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print('Provided invalid input parameters, needs to be ([path to config file])')
		exit(1)
	config_uri = sys.argv[1]

	# Read configuration file
	with open(config_uri, mode='r') as f:
		config_data = json.loads(f.read())

	# Get model selector config parameters
	num_input_neurons = int(config_data['num_input_neurons'])
	num_output_neurons = int(config_data['num_output_neurons'])
	hidden_layer_spaces = config_data['hidden_layer_spaces']
	num_examples = int(config_data['num_examples'])
	k_cross_folds = int(config_data['k_cross_folds'])
	dropout_rate_space = config_data['dropout_rate_space']
	num_iterations = int(config_data['num_iterations'])
	batch_size = int(config_data['batch_size'])

	# Get system config parameters
	data_reader_uri = config_data['data_reader_uri']
	data_uri = config_data['data_uri']
	log_uri = config_data['log_uri']
	results_uri = config_data['results_uri']

	# Create the parameters needed to setup multiple scenarios of ANN archictures
	parameter_sets = create_parameter_sets(hidden_layer_spaces, dropout_rate_space)

	# Read training and testing data
	print('Loading data...')
	features, _, _, labels, _, _ = read_numerai(data_uri, num_examples, 0, 1)
	print('Data loaded!')
	print

	# Determine the k-fold cross validation indicies for training and testing
	k_folds_idx = KFold(num_examples, n_folds=k_cross_folds)

	# Normalize features
	for i in range(features.shape[1]):
		features[:,i] = (features[:,i] - np.mean(features[:,i])) / np.std(features[:,i])

	# Initialize results
	# TODO: Use itertools to preset repeatable values
	results = {
		'testing_log_loss': [],
		'testing_accuracy': [],
		'training_log_loss': [],
		'training_accuracy': [],
		'num_layers': [],
		'num_neurons': [],
		'dropout_rates': [],
		'k_cross_folds': [],
		'num_examples': [],
		'num_iterations': [],
		'batch_size': [],
	}

	# Start TensorFlow session
	sess = tf.InteractiveSession()

	# Evaluate each ANN parameter set
	n = 1
	for parameter_set in parameter_sets:
		print('Running scenario %d/%d' % (n, len(parameter_sets)))
		hidden_layers = parameter_set['hidden_layers']
		dropout_rates = parameter_set['dropout_rates']
		num_hidden_layers = len(hidden_layers)
		num_dropout_rates = len(dropout_rates)

		# Setup neuron-layer
		neurons = [num_input_neurons, num_output_neurons]
		if(parameter_set['hidden_layers']):
			for i in range(num_hidden_layers):
				neurons.insert(i+1, parameter_set['hidden_layers'][i])
		else:
			raise NotImplementedError('Logistic regression is not yet supported.')

		# Create ANN
		ann = ArtificialNeuralNetworkClassifier(sess, neurons, dropout_rates)

		# Setup k-fold cross validation results
		kf_results = {}
		kf_results['training_log_loss'] = []
		kf_results['training_accuracy'] = []
		kf_results['testing_log_loss'] = []
		kf_results['testing_accuracy'] = []

		# Train and predict k times
		for training_idx, testing_idx in k_folds_idx:
			ann.train(sess, features[training_idx], labels[training_idx], num_iterations, batch_size, False)

			# Compute the training accuracies
			kf_results['training_log_loss'].append(ann.log_loss(sess, features[training_idx], labels[training_idx]))
			kf_results['training_accuracy'].append(ann.matches(sess, features[training_idx], labels[training_idx]))

			# Compute the testing accuracies
			kf_results['testing_accuracy'].append(ann.log_loss(sess, features[testing_idx], labels[testing_idx]))
			kf_results['testing_log_loss'].append(ann.matches(sess, features[testing_idx], labels[testing_idx]))

		# Store results for later analysis
		results['training_log_loss'].append(np.mean(kf_results['training_log_loss']))
		results['training_accuracy'].append(np.mean(kf_results['training_accuracy']))
		results['testing_log_loss'].append(np.mean(kf_results['testing_log_loss']))
		results['testing_accuracy'].append(np.mean(kf_results['testing_accuracy']))
		results['num_layers'].append(num_hidden_layers)
		results['num_neurons'].append(hidden_layers)
		results['dropout_rates'].append(dropout_rates)
		results['k_cross_folds'].append(k_cross_folds)
		results['num_examples'].append(num_examples)
		results['num_iterations'].append(num_iterations)
		results['batch_size'].append(batch_size)

		n += 1

	# Output results
	print
	print('Storing results...')
	df = pd.DataFrame(results)
	df.to_csv("results/ann_model_selector_results_%s.csv" % datetime.now(), index=False)
	print('Results stored!')
	print
