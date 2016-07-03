import os
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import dropbox
from ArtificialNeuralNetworkClassifier import ArtificialNeuralNetworkClassifier
from datetime import datetime
from read_numerai import *

DROPBOX_AUTH_TOKEN = 'qECz4Lio64gAAAAAAAADKCBiIafW0-teoaxb7jaNJVjcn517S7mH0l7rwjZXbThX'


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print('Provided invalid input parameters, needs to be ([config filename] [verbose?])')
		exit(1)
	config_filename = sys.argv[1]
	verbose = int(sys.argv[2])

	# Initialize the Dropbox client
	client = dropbox.client.DropboxClient(DROPBOX_AUTH_TOKEN)
	
	# Download configuration file from Dropbox
	print('Downloading configuration file...')
	f, metadata = client.get_file_and_metadata('config/' + config_filename)
	print('Configuration file downloaded!')
	print
	config_data = json.loads(f.read())

	# Get model selector parameters
	num_input_neurons = int(config_data['num_input_neurons'])
	num_output_neurons = int(config_data['num_output_neurons'])
	num_examples = int(config_data['num_examples'])
	hidden_layers = config_data['hidden_layers']
	dropout_rates = config_data['dropout_rates']
	num_iterations = int(config_data['num_iterations'])
	batch_size = int(config_data['batch_size'])
	num_hidden_layers = len(hidden_layers)

	# Get system parameters
	data_reader_uri = config_data['data_reader_uri']
	data_dir = config_data['data_dir']
	training_filename = config_data['training_filename']
	tournament_filename = config_data['tournament_filename']

	# Check if the data directory exists
	if(not os.path.isdir("data")):
		os.makedirs("data")

	# Before pulling data, check if it already exists locally
	local_data_uri = "data/" + data_dir + "/"
	train_exists = os.path.exists(local_data_uri + training_filename)
	tourney_exists = os.path.exists(local_data_uri + tournament_filename)
	if(not train_exists or not tourney_exists):
		os.makedirs(local_data_uri)

		# Download data locally from Dropbox
		print('Downloading data...')
		f, metadata = client.get_file_and_metadata(local_data_uri + training_filename)
		out = open(local_data_uri + training_filename, 'w')
		out.write(f.read())
		out.close()
		f, metadata = client.get_file_and_metadata(local_data_uri + tournament_filename)
		out = open(local_data_uri + tournament_filename, 'w')
		out.write(f.read())
		out.close()
		print('Data downloaded!')
		print

	# Read training and testing data
	print('Loading data...')
	train_features, _, tourney_features, train_labels, _, tourney_ids = read_numerai(local_data_uri, num_examples, 0, 1)
	print('Data loaded!')
	print

	# Normalize features
	for i in xrange(train_features.shape[1]):
		train_features[:,i] = (train_features[:,i] - np.mean(train_features[:,i])) / np.std(train_features[:,i])
	for i in xrange(tourney_features.shape[1]):
		tourney_features[:,i] = (tourney_features[:,i] - np.mean(tourney_features[:,i])) / np.std(tourney_features[:,i])

	# Start TensorFlow session
	if(verbose):
		sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
	else:
		sess = tf.InteractiveSession()

	# Setup neurons-layers
	neurons = [num_input_neurons, num_output_neurons]
	if(hidden_layers):
		for i in xrange(num_hidden_layers):
			neurons.insert(i+1, hidden_layers[i])
	else:
		raise NotImplementedError('Logistic regression is not yet supported.')

	# Create and train the ANN
	ann = ArtificialNeuralNetworkClassifier(sess, neurons, dropout_rates)
	ann.train(sess, train_features, train_labels, num_iterations, batch_size, True)

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

	# Upload results to Dropbox
	print('Uploading results...')
	f = open(results_file_uri, 'r')
	response = client.put_file(results_file_uri, f)
	print('Results uploaded!')
	print
