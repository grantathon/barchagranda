import tensorflow as tf


class ArtificialNeuralNetwork(object):
	def __init__(self, neurons, dropout_rates):
		if((neurons - 2) < 0):
			raise ValueError('There must be at least an input and output layer.')

		# Initialize data members
		self.num_input_neurons = neurons[0]
		self.hidden_neurons = neurons[1:-2]
		self.num_output_neurons = neurons[-1]
		self.dropout_rates = dropout_rates
		self.num_dropout_rates = len(dropout_rates)
		self.num_hidden_layers = len(self.hidden_neurons)

		# pprint(self.num_input_neurons)
		# pprint(self.num_output_neurons)
		# pprint(self.num_hidden_neurons)
		# pprint(self.dropout_rates)
		# pprint(self.num_hidden_layers)

		if(self.num_hidden_layers != len(dropout_rates)):
			raise ValueError('The number of dropout rates and hidden layers must be equal.')

		self.__create_ann()

	def train(self, features, labels, num_iterations, batch_size, verbose=False):
		# Iteratively train the ANN
		for i in range(num_iterations):
			# Periodically evaluate accuracy
			if(verbose and i % 1000 == 0):
				train_log_loss = self.cross_entropy.eval(feed_dict={x: features, y_: labels})
				# train_log_loss = self.cross_entropy.eval(feed_dict={x: train_features, y_: train_labels})
				print("Step %d, training log loss %.5f" % (i, train_log_loss))

			# Generate random indices and run training step
			idx = random.sample(xrange(len(features)-1), batch_size)
			self.train_step.run(feed_dict={x: features[idx], y_: labels[idx]})

	def predict(self, features):
		return self.y.eval(feed_dict={x: features})

	def log_loss(self, features, labels):
		eval_drs = [1.0 for i in range(self.num_dropout_rates)]
		return self.cross_entropy.eval(feed_dict={x: features, y_: labels, dr: eval_drs})

	def accuracy(self, features, labels):
		eval_drs = [1.0 for i in range(self.num_dropout_rates)]
		return self.accuracy.eval(feed_dict={x: features, y_: labels, dr: eval_drs}))

	def __create_ann(self):
		# Placeholders for training examples and labels
		self.x = tf.placeholder(tf.float32, [None, self.num_input_neurons])
		self.y_ = tf.placeholder(tf.float32, [None, self.num_output_neurons])

		# Placeholder for dropout rates
		self.dr = tf.placeholder(tf.float32, [self.num_dropout_rates])

		# Construct the ANN layer by layer
		self.W = []
		self.b = []
		self.h = []
		self.h_drop = []
		for i in range(self.num_hidden_layers):
			# Determine proper dimensions for weight matrix and bias vector
			if(i != 0):
				input_count = layer_neurons[i-1]
			else:
				input_count = self.num_input_neurons
			output_count = layer_neurons[i]

			# Create weight matrix and bias vector
			self.W.append(weight_variable([input_count, output_count]))
			self.b.append(bias_variable([output_count]))

			# Perform dropout on hidden layer
			self.h_drop.append(tf.nn.dropout(h[i], dr[i]))
			
			# Create hidden layer
			if(i != 0):
				self.h.append(tf.nn.relu(tf.matmul(h_drop[i-1], W[i]) + b[i]))
			else:
				self.h.append(tf.nn.relu(tf.matmul(x, W[i]) + b[i]))

			# Check if hidden layer is connected to output layer
			if(i == self.num_hidden_layers-1):
				# Create weight matrix and bias vector
				self.W.append(weight_variable([output_count, output_neuron_count]))
				self.b.append(bias_variable([output_neuron_count]))

				# Create output layer
				self.y = tf.nn.softmax(tf.matmul(h_drop[i], W[i+1]) + b[i+1])

		# Implement cross-entropy (i.e., log loss) as loss function
		self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

		# Initialize optimizer
		# self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

		# Determine the accuracy of the model
		self.correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
