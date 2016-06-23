import tensorflow as tf
import pandas as pd
import numpy as np
import json
import random
from sklearn.cross_validation import KFold
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
	layer_count = int(config_data['layer_count'])
	input_neuron_count = int(config_data['input_neuron_count'])
	output_neuron_count = int(config_data['output_neuron_count'])
	example_count = int(config_data['example_count'])
	k_cross_folds = int(config_data['k_cross_folds'])
	dropout_rate_space = config_data['dropout_rate_space']
	iteration_count = int(config_data['iteration_count'])
	batch_size = int(config_data['batch_size'])

	# Get system parameters
	data_reader_uri = config_data['data_reader_uri']
	data_uri = config_data['data_uri']
	log_uri = config_data['log_uri']
	results_uri = config_data['results_uri']

	# Read training and testing data
	print('Loading data...')
	features, _, _, labels, _, _ = read_numerai(data_uri, example_count, 0, 1)
	print('Data loaded!')
	print

	# Determine the k-fold cross validation indicies for training and testing
	k_folds_idx = KFold(example_count, n_folds=k_cross_folds)

	# Determine layer-neuron and dropout rate scenarios
	# TODO: Make this recursive
	layer_neuron_scenarios = []
	dropout_rate_scenarios = []
	dropout_rates = np.linspace(dropout_rate_space[0], dropout_rate_space[1], dropout_rate_space[2])
	for i in range(layer_count):
		layer_neuron_scenarios.append([])
		dropout_rate_scenarios.append([])
		
		if(i == 0):
			for j in range(output_neuron_count, input_neuron_count+1):
				layer_neuron_scenarios[i].append([j])
			for j in range(len(dropout_rates)):
				dropout_rate_scenarios[i].append([round(dropout_rates[j], 4)])
		elif(i == 1):
			for j in range(output_neuron_count, input_neuron_count+1):
				for k in range(output_neuron_count, j+1):
					layer_neuron_scenarios[i].append([j, k])
			for j in range(len(dropout_rates)):
				for k in range(len(dropout_rates)):
					dropout_rate_scenarios[i].append([round(dropout_rates[j], 4), round(dropout_rates[k], 4)])
		elif(i == 2):
			for j in range(output_neuron_count, input_neuron_count+1):
				for k in range(output_neuron_count, j+1):
					for l in range(output_neuron_count, k+1):
						layer_neuron_scenarios[i].append([j, k, l])
			for j in range(len(dropout_rates)):
				for k in range(len(dropout_rates)):
					for l in range(len(dropout_rates)):
						dropout_rate_scenarios[i].append([round(dropout_rates[j], 4), round(dropout_rates[k], 4), \
							round(dropout_rates[l], 4)])
		else:
			raise NotImplementedError('Does not support %i hidden layers' % (i+1))

	# Determine number of total scenarios to run
	total_layer_neuron_scenarios = np.sum([len(layer_neuron_scenarios[i]) for i in range(layer_count)])
	total_dropout_scenarios = np.sum([len(dropout_rate_scenarios[i]) for i in range(layer_count)])
	total_scenarios = total_layer_neuron_scenarios * total_dropout_scenarios

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
		'layer_count': [],
		'neuron_counts': [],
		'dropout_rates': [],
		'k_cross_folds': [],
		'example_count': [],
		'iteration_count': [],
		'batch_size': [],
	}

	# Start TensorFlow session
	sess = tf.InteractiveSession()

	# Evaluate all possible scenarios
	# TODO: Try to extract some session placeholders, variables, and other things
	n = 1
	for i in range(layer_count):
		for dropout_rate_scenario in dropout_rate_scenarios[i]:
			drs = [dropout_rate_scenario[j] for j in range(i+1)]
			eval_drs = [1.0 for i in range(len(drs))]

			for layer_neuron_scenario in layer_neuron_scenarios[i]:
				print('Running scenario %d/%d' % (n, total_scenarios))
				n += 1

				layer_neurons = [layer_neuron_scenario[j] for j in range(i+1)]

				# Placeholders for training examples and labels
				x = tf.placeholder(tf.float32, [None, input_neuron_count])
				y_ = tf.placeholder(tf.float32, [None, output_neuron_count])

				# Placeholder for dropout rates
				dr = tf.placeholder(tf.float32, [len(drs)])

				# Construct the ANN layer by layer
				W = []
				b = []
				h = []
				h_drop = []
				for j in range(i+1):
					# Determine proper dimensions for weight matrix and bias vector
					if(j == 0):
						input_count = input_neuron_count
					else:
						input_count = layer_neurons[j-1]
					output_count = layer_neurons[j]

					# Create weight matrix and bias vector
					W.append(weight_variable([input_count, output_count]))
					b.append(bias_variable([output_count]))
					
					# Create hidden layer
					h.append(tf.nn.relu(tf.matmul(x, W[j]) + b[j]))

					# Perform dropout on hidden layer
					h_drop.append(tf.nn.dropout(h[j], dr[j]))

					# Check if hidden layer is connected to output layer
					if(j == i):
						# Create weight matrix and bias vector
						W.append(weight_variable([output_count, output_neuron_count]))
						b.append(bias_variable([output_neuron_count]))

						# Create output layer
						y = tf.nn.softmax(tf.matmul(h_drop[j], W[j+1]) + b[j+1])

				# Implement cross-entropy (i.e., log loss) as loss function
				cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

				# Initialize optimizer
				# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
				train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

				# Determine the accuracy of the model
				correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

				# Initialize variables
				sess.run(tf.initialize_all_variables())

				# Setup k-fold cross validation results
				kf_results = {}
				kf_results['training_log_loss'] = []
				kf_results['training_accuracy'] = []
				kf_results['testing_log_loss'] = []
				kf_results['testing_accuracy'] = []

				# Train and test ANN k times
				for training_idx, testing_idx in k_folds_idx:
					# Iteratively train the ANN
					for j in range(iteration_count):
						# Generate a batch of random training indices and run training step
						idx = random.sample(training_idx, batch_size)
						train_step.run(feed_dict={x: features[idx], y_: labels[idx], dr: drs})

					# Compute the training accuracies
					kf_results['training_log_loss'].append(cross_entropy.eval(feed_dict={x: features[training_idx], \
						y_: labels[training_idx], dr: eval_drs}))
					kf_results['training_accuracy'].append(accuracy.eval(feed_dict={x: features[training_idx], \
						y_: labels[training_idx], dr: eval_drs}))

					# Compute the testing accuracies
					kf_results['testing_accuracy'].append(accuracy.eval(feed_dict={x: features[testing_idx], \
						y_: labels[testing_idx], dr: eval_drs}))
					kf_results['testing_log_loss'].append(cross_entropy.eval(feed_dict={x: features[testing_idx], \
						y_: labels[testing_idx], dr: eval_drs}))

				# Store results for later analysis
				results['training_log_loss'].append(np.mean(kf_results['training_log_loss']))
				results['training_accuracy'].append(np.mean(kf_results['training_accuracy']))
				results['testing_log_loss'].append(np.mean(kf_results['testing_log_loss']))
				results['testing_accuracy'].append(np.mean(kf_results['testing_accuracy']))
				results['layer_count'].append(i+1)
				results['neuron_counts'].append(layer_neurons)
				results['dropout_rates'].append(drs)
				results['k_cross_folds'].append(k_cross_folds)
				results['example_count'].append(example_count)
				results['iteration_count'].append(iteration_count)
				results['batch_size'].append(batch_size)

	# Output results
	print
	print('Storing results...')
	df = pd.DataFrame(results)
	df.to_csv("results/ann_model_selector_results_%s.csv" % datetime.now(), index=False)
	print('Results stored!')
	print
