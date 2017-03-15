import os
import tensorflow as tf
import pandas as pd
import numpy as np
import json
from pprint import pprint
from ArtificialNeuralNetworkClassifier import ArtificialNeuralNetworkClassifier
from datetime import datetime
from read_numerai import *


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print('Provided invalid input parameters, needs to be ([config filename] [verbose?])')
		exit(1)
	config_uri = sys.argv[1]
	verbose = int(sys.argv[2])

	# Read the config file
	with open(config_uri) as f:
		config_data = json.load(f)
	# pprint(config_data)
	# exit(0)

	# Get model selector parameters
	num_input_neurons = int(config_data['num_input_neurons'])
	num_output_neurons = int(config_data['num_output_neurons'])
	num_examples = int(config_data['num_examples'])
	hidden_layers = config_data['hidden_layers']
	dropout_rates = config_data['dropout_rates']
	max_iterations = int(config_data['max_iterations'])
	batch_size = int(config_data['batch_size'])
	num_hidden_layers = len(hidden_layers)
	data_uri = config_data["data_uri"]

	# Read training and testing data
	print('Loading data...')
	train_features, _, tourney_features, train_labels, _, tourney_ids = read_numerai(data_uri, num_examples, 0, 1)
	print('Data loaded!')
	print

	# pprint(train_features)
	# pprint(tourney_features)
	# pprint(train_labels)
	# pprint(tourney_ids)
	# exit(0)

	# Normalize features
	for i in range(train_features.shape[1]):
		train_features[:,i] = (train_features[:,i] - np.mean(train_features[:,i])) / np.std(train_features[:,i])
	for i in range(tourney_features.shape[1]):
		tourney_features[:,i] = (tourney_features[:,i] - np.mean(tourney_features[:,i])) / np.std(tourney_features[:,i])

	# Start TensorFlow session
	if(verbose):
		sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
	else:
		sess = tf.InteractiveSession()

	# Setup neurons-layers
	neurons = [num_input_neurons, num_output_neurons]
	if(hidden_layers):
		for i in range(num_hidden_layers):
			neurons.insert(i+1, hidden_layers[i])
	else:
		raise NotImplementedError('Logistic regression is not yet supported.')

	# Create and train the ANN
	ann = ArtificialNeuralNetworkClassifier(sess, neurons, dropout_rates)
	ann.train(sess, train_features, train_labels, max_iterations, batch_size, True)

	# Display final log loss
	print('Final training log loss: %.5f' % ann.log_loss(sess, train_features, train_labels))
	print

	# Make predictions on unlabeled (tournament) examples
	probabilities = ann.predict(sess, tourney_features)

	# Check if the data directory exists
	if(not os.path.isdir("results")):
		os.makedirs("results")

	# Create a data frame with the results and save locally
	print('Storing results...')
	idx = range(0, len(tourney_ids))
	d = {'t_id': pd.Series(tourney_ids, idx), 'probability': pd.Series(probabilities[:,1], idx)}
	df = pd.DataFrame(d)
	results_file_uri = "results/ann_model_runner_results_%s.csv" % datetime.now()
	df.to_csv(results_file_uri, columns=['t_id', 'probability'], index=False)
	print('Results stored!')
	print
