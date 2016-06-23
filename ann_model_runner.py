import tensorflow as tf
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime
from pprint import pprint
from read_numerai import *


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print('Provided invalid input parameters, needs to be ([path to config file])')
		exit(1)
	config_uri = sys.argv[1]

	# Read configuration file
	with open(config_uri, mode='r') as f:
		config_data = json.loads(f.read())

	# Get model selector parameters
	input_neuron_count = int(config_data['input_neuron_count'])
	output_neuron_count = int(config_data['output_neuron_count'])
	example_count = int(config_data['example_count'])
	layer_neurons = config_data['layer_neurons']
	dropout_rates = config_data['dropout_rates']
	iteration_count = int(config_data['iteration_count'])
	batch_size = int(config_data['batch_size'])
	layer_count = len(layer_neurons)

	# Get system parameters
	data_reader_uri = config_data['data_reader_uri']
	data_uri = config_data['data_uri']
	log_uri = config_data['log_uri']
	results_uri = config_data['results_uri']

	# Read training and testing data
	print('Loading data...')
	train_features, _, tourney_features, train_labels, _, tourney_ids = read_numerai(data_uri, example_count, 0, 1)
	print('Data loaded!')
	print

	# Normalize features
	for i in range(train_features.shape[1]):
		train_features[:,i] = (train_features[:,i] - np.mean(train_features[:,i])) / np.std(train_features[:,i])
	for i in range(tourney_features.shape[1]):
		tourney_features[:,i] = (tourney_features[:,i] - np.mean(tourney_features[:,i])) / np.std(tourney_features[:,i])

	# Start TensorFlow session
	sess = tf.InteractiveSession()

	# Placeholders for training examples and labels
	x = tf.placeholder(tf.float32, [None, input_neuron_count])
	y_ = tf.placeholder(tf.float32, [None, output_neuron_count])

	# Variable for dropout rates
	dr = tf.Variable(dropout_rates)

	# Construct the ANN layer by layer
	W = []
	b = []
	h = []
	h_drop = []
	for i in range(layer_count):
		# Determine proper dimensions for weight matrix and bias vector
		if(i == 0):
			input_count = input_neuron_count
		else:
			input_count = layer_neurons[i-1]
		output_count = layer_neurons[i]

		# Perform dropout on hidden layer
		h_drop.append(tf.nn.dropout(h[i], dr[i]))

		# Create weight matrix and bias vector
		W.append(weight_variable([input_count, output_count]))
		b.append(bias_variable([output_count]))

		# Create hidden layer
		if(j != 0):
			h.append(tf.nn.relu(tf.matmul(h_drop[j-1], W[j]) + b[j]))
		else:
			h.append(tf.nn.relu(tf.matmul(x, W[j]) + b[j]))

		# Check if hidden layer is connected to output layer
		if(i == layer_count-1):
			# Create weight matrix and bias vector
			W.append(weight_variable([output_count, output_neuron_count]))
			b.append(bias_variable([output_neuron_count]))

			# Create output layer
			y = tf.nn.softmax(tf.matmul(h_drop[i], W[i+1]) + b[i+1])

	# Implement cross-entropy (i.e., log loss) as loss function
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

	# Initialize optimizer
	# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	# Initialize variables
	sess.run(tf.initialize_all_variables())

	# Iteratively train the ANN
	for i in range(iteration_count):
		# Periodically evaluate accuracy
		if(i % 1000 == 0):
			train_log_loss = cross_entropy.eval(feed_dict={x: train_features, y_: train_labels})
			print("Step %d, training log loss %.5f" % (i, train_log_loss))

		# Generate random indices and run training step
		idx = random.sample(xrange(len(train_features)-1), batch_size)
		train_step.run(feed_dict={x: train_features[idx], y_: train_labels[idx]})

	# Display final log loss
	print('Final training log loss: %.5f' % cross_entropy.eval(feed_dict={x: train_features, y_: train_labels}))

	# Make predictions on unlabeled (tournament) examples
	probabilities = y.eval(feed_dict={x: tourney_features})

	# Create a data frame with the results and save
	print
	print('Storing results...')
	idx = range(0, len(tourney_ids))
	d = {'t_id': pd.Series(tourney_ids, idx), 'probability': pd.Series(probabilities[:,1], idx)}
	df = pd.DataFrame(d)
	df.to_csv("results/ann_model_runner_results_%s.csv" % datetime.now(), columns=['t_id', 'probability'], index=False)
	print('Results stored!')
	print
