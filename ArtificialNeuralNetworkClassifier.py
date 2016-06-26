import tensorflow as tf
import random


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
		# Iteratively train the ANN
		for i in range(num_iterations):
			# Periodically evaluate accuracy
			if(verbose and i % 100 == 0):
				eval_drs = [1.0 for j in range(self.num_dropout_rates)]
				train_log_loss = self.cross_entropy.eval(feed_dict={self.x: features, self.y_: labels, self.dr: eval_drs})
				print("Step %d, training log loss %.5f" % (i, train_log_loss))

			# Generate random indices and run training step
			idx = random.sample(xrange(len(features)-1), batch_size)
			self.train_step.run(feed_dict={self.x: features[idx], self.y_: labels[idx], self.dr: self.dropout_rates})

	def predict(self, session, features):
		eval_drs = [1.0 for i in range(self.num_dropout_rates)]
		return self.y.eval(feed_dict={self.x: features, self.dr: eval_drs})

	def log_loss(self, session, features, labels):
		eval_drs = [1.0 for i in range(self.num_dropout_rates)]
		return self.cross_entropy.eval(feed_dict={self.x: features, self.y_: labels, self.dr: eval_drs})

	def matches(self, session, features, labels):
		eval_drs = [1.0 for i in range(self.num_dropout_rates)]
		return self.accuracy.eval(feed_dict={self.x: features, self.y_: labels, self.dr: eval_drs})

	def __weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def __bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
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
		for i in range(self.num_hidden_layers):
			# Determine proper dimensions for weight matrix and bias vector
			if(i != 0):
				input_count = self.layer_neurons[i-1]
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
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

		# Determine the accuracy of the model
		self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

		# Initialize variables
		session.run(tf.initialize_all_variables())