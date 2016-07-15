import tensorflow as tf
import numpy as np


class ArtificialNeuralNetworkClassifier(object):
	def __init__(self, session, neurons, dropout_rates):
		if((len(neurons) - 2) < 0):
			raise ValueError('There must be at least an input and output layer.')

		# Initialize data members
		self.layer_neurons = neurons
		self.num_input_neurons = neurons[0]
		self.hidden_neurons = neurons[1:len(neurons)-1]
		self.num_output_neurons = neurons[-1]
		self.dropout_rates = dropout_rates
		self.num_dropout_rates = len(dropout_rates)
		self.num_hidden_layers = len(self.hidden_neurons)

		if(self.num_hidden_layers != self.num_dropout_rates):
			raise ValueError('The number of dropout rates and hidden layers must be equal.')

		self.__create_ann(session)

	def train(self, session, features, labels, num_iterations, batch_size, verbose=False):
		num_examples = len(features)
		num_batches = int(np.ceil(num_examples/float(batch_size)))

		# Iteratively train the ANN for multiple epochs
		for i in xrange(num_iterations):
			# Shuffle training set into seperate batches
			indices = np.random.permutation(num_examples)

			# Run training step per batch
			for j in xrange(num_batches):
				batch_indices = indices[j*batch_size:(j+1)*batch_size]
				self.train_step.run(feed_dict={self.x: features[batch_indices], self.y_: labels[batch_indices],
					self.dr: self.dropout_rates})

			# Periodically evaluate accuracy
			if(verbose and i % 100 == 0):
				print("Epoch step %d, training log loss %.5f" % (i, self.log_loss(session, features, labels)))

	def predict(self, session, features):
		eval_drs = [1.0 for i in xrange(self.num_dropout_rates)]
		return self.y.eval(feed_dict={self.x: features, self.dr: eval_drs})

	def log_loss(self, session, features, labels):
		eval_drs = [1.0 for i in xrange(self.num_dropout_rates)]
		return self.cross_entropy.eval(feed_dict={self.x: features, self.y_: labels, self.dr: eval_drs})

	def matches(self, session, features, labels):
		eval_drs = [1.0 for i in xrange(self.num_dropout_rates)]
		return self.accuracy.eval(feed_dict={self.x: features, self.y_: labels, self.dr: eval_drs})

	# @staticmethod
	# def eval_one_layer_ann(features, labels, num_iterations, batch_size, k_folds, num_input_neurons, num_output_neurons,
	# 	dropout_rate0, hidden_layer0):
	# 	# Start session
	# 	sess = tf.InteractiveSession()
		
	# 	# Setup neuron and dropout rates structures
	# 	neurons = [num_input_neurons, hidden_layer0, num_output_neurons]
	# 	dropout_rates = [dropout_rate0]

	# 	# Create the ANN
	# 	ann = ArtificialNeuralNetworkClassifier(sess, neurons, dropout_rates)

	# 	# Determine the k-fold cross validation indicies for training and testing
	# 	k_folds_idx = KFold(len(features), n_folds=k_folds)

	# 	# TODO: Perform k-fold cross validation
	# 	log_loss = []
	# 	for training_idx, testing_idx in k_folds_idx:
	# 		# Train the ANN
	# 		ann.train(sess, features[training_idx], labels[training_idx], num_iterations, batch_size, False)

	# 		# Evaluate and save the ANN's accuracy
	# 		log_loss.append(ann.log_loss(sess, features[testing_idx], labels[testing_idx]))

	# 	# Return the average log loss
	# 	return log_loss.mean()

	# @staticmethod
	# def eval_two_layer_ann(self, dropout_rate0, dropout_rate1, hidden_layer0, hidden_layer1):
	# 	pass

	def __weight_variable(self, shape):
		initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
		return tf.Variable(initial)

	def __bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		# initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
		return tf.Variable(initial)

	def __create_ann(self, session):
		# Placeholders for training examples and labels
		self.x = tf.placeholder(tf.float32, [None, self.num_input_neurons])
		self.y_ = tf.placeholder(tf.float32, [None, self.num_output_neurons])

		# Placeholder for dropout rates
		self.dr = tf.placeholder(tf.float32, [self.num_dropout_rates])

		# Construct the ANN layer by layer
		W = []
		b = []
		h = []
		h_drop = []
		for i in xrange(self.num_hidden_layers):
			# Determine proper dimensions for weight matrix and bias vector
			if(i != 0):
				input_count = self.layer_neurons[i]
			else:
				input_count = self.num_input_neurons
			output_count = self.hidden_neurons[i]

			# Create weight matrix and bias vector
			W.append(self.__weight_variable([input_count, output_count]))
			b.append(self.__bias_variable([output_count]))

			# Create hidden layer
			if(i != 0):
				h.append(tf.nn.relu(tf.matmul(h_drop[i-1], W[i]) + b[i]))
			else:
				h.append(tf.nn.relu(tf.matmul(self.x, W[i]) + b[i]))

			# Perform dropout on hidden layer
			h_drop.append(tf.nn.dropout(h[i], self.dr[i]))
			
			# Check if hidden layer is connected to output layer
			if(i == self.num_hidden_layers-1):
				# Create weight matrix and bias vector
				W.append(self.__weight_variable([output_count, self.num_output_neurons]))
				b.append(self.__bias_variable([self.num_output_neurons]))

				# Create output layer
				self.y = tf.nn.softmax(tf.matmul(h_drop[i], W[i+1]) + b[i+1])

		# Implement cross-entropy (i.e., log loss) as loss function
		self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_*tf.log(self.y), reduction_indices=[1]))

		# Initialize optimizer
		# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
		self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.cross_entropy)

		# Determine the accuracy of the model
		self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

		# Initialize variables
		session.run(tf.initialize_all_variables())
