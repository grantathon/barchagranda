import tensorflow as tf
import numpy as np
from collections import deque


class ArtificialNeuralNetworkClassifier(object):
	def __init__(self, session, neurons, dropout_rates, learning_rate, regularization_param, batch_size):
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
		self.learning_rate = learning_rate
		self.regularization_param = regularization_param
		self.batch_size = batch_size

		if(self.num_hidden_layers != self.num_dropout_rates):
			raise ValueError('The number of dropout rates and hidden layers must be equal.')

		self.__create_ann(session)

	def train(self, session, features, labels, max_iterations, validation_percent, verbose=False):
		num_examples = len(features)
		num_validate_examples = int(num_examples * validation_percent)
		num_train_examples = num_examples - num_validate_examples
		num_batches = int(np.ceil(num_train_examples / float(self.batch_size)))

		# Separate validation examples
		validation_features = features[num_train_examples:]
		validation_labels = labels[num_train_examples:]

		# Iteratively train the ANN for multiple epochs
		#avg_periods = 100
		#log_losses = deque(maxlen=avg_periods)
		for i in xrange(max_iterations):
			# Shuffle training set into seperate batches
			indices = np.random.permutation(num_train_examples)

			# Run training step per batch
			for j in xrange(num_batches):
				batch_indices = indices[(j * self.batch_size):((j + 1) * self.batch_size)]
				self.train_step.run(feed_dict={self.x: features[batch_indices], self.y_: labels[batch_indices],
					self.dr: self.dropout_rates, self.eta: self.learning_rate, self.beta: self.regularization_param})

			# Early stop
			#log_loss = self.log_loss(session, validation_features, validation_labels)
			#log_losses.append(log_loss)
			#if(i >= avg_periods - 1 and np.mean(log_losses) < log_loss):
			#	print("Early stop occurred at epoch %d, log loss %.5f" % (i, log_loss))
			#	break;

			# Periodically evaluate accuracy
			if(verbose and i % 10 == 0):
				#print("Epoch step %d, log loss %.5f" % (i, log_loss))
				log_loss = self.log_loss(session, validation_features, validation_labels)
				accuracy = self.matches(session, validation_features, validation_labels)
				print("Epoch step %d, log loss %.5f, accuracy %.5f" % (i, log_loss, accuracy))

	def predict(self, session, features):
		eval_drs = [1.0 for i in xrange(self.num_dropout_rates)]
		return self.y.eval(feed_dict={self.x: features, self.dr: eval_drs, self.eta: self.learning_rate,
			self.beta: self.regularization_param})

	def log_loss(self, session, features, labels):
		eval_drs = [1.0 for i in xrange(self.num_dropout_rates)]
		return self.cross_entropy.eval(feed_dict={self.x: features, self.y_: labels, self.dr: eval_drs,
			self.eta: self.learning_rate, self.beta: self.regularization_param})

	def matches(self, session, features, labels):
		eval_drs = [1.0 for i in xrange(self.num_dropout_rates)]
		return self.accuracy.eval(feed_dict={self.x: features, self.y_: labels, self.dr: eval_drs,
			self.eta: self.learning_rate, self.beta: self.regularization_param})

	def __weight_variable(self, shape):
		initial = tf.truncated_normal(shape, mean=0.0, stddev=(1 / np.sqrt(shape[0])))
		return tf.Variable(initial)

	def __bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def __create_ann(self, session):
		# Placeholders for training examples and labels
		self.x = tf.placeholder(tf.float32, [None, self.num_input_neurons])
		self.y_ = tf.placeholder(tf.float32, [None, self.num_output_neurons])

		# Placeholder for dropout rates and regularization parameter
		self.dr = tf.placeholder(tf.float32, [self.num_dropout_rates])
		self.eta = tf.placeholder(tf.float32, [])
		self.beta = tf.placeholder(tf.float32, [])

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
				h.append(tf.nn.relu(tf.matmul(h_drop[i - 1], W[i]) + b[i]))
			else:
				h.append(tf.nn.relu(tf.matmul(self.x, W[i]) + b[i]))

			# Perform dropout on hidden layer
			h_drop.append(tf.nn.dropout(h[i], self.dr[i]))
			
			# Check if hidden layer is connected to output layer
			if(i == self.num_hidden_layers - 1):
				# Create weight matrix and bias vector
				W.append(self.__weight_variable([output_count, self.num_output_neurons]))
				b.append(self.__bias_variable([self.num_output_neurons]))

				# Create output layer
				self.y = tf.matmul(h_drop[i], W[i + 1]) + b[i + 1]

		# Implement cross-entropy (i.e., log loss) as loss function
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
		regularizers = tf.nn.l2_loss(W[0])
		for w in W[1:]:
			regularizers += tf.nn.l2_loss(w)
		self.cross_entropy = tf.reduce_mean(loss + self.beta * regularizers)

		# Initialize optimizer
		#self.train_step = tf.train.GradientDescentOptimizer(self.eta).minimize(self.cross_entropy)
		self.train_step = tf.train.AdamOptimizer(self.eta).minimize(self.cross_entropy)

		# Determine the accuracy of the model
		self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

		# Initialize variables
		session.run(tf.initialize_all_variables())
