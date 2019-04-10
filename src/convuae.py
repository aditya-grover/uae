from utils import *
import tensorflow as tf 
import numpy as np
import time
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

class CONVUAE():
	# convolutional uncertainty autoencoder

	def __init__(self, sess, datasource, vae=False):

		self.seed = FLAGS.seed
		tf.set_random_seed(self.seed)
		np.random.seed(self.seed)

		self.sess = sess
		self.datasource = datasource
		
		self.input_dim = self.datasource.input_dim
		self.input_height = self.datasource.input_height
		self.input_width = self.datasource.input_width
		self.input_channels = self.datasource.input_channels

		self.z_dim = FLAGS.num_measurements
		self.last_layer_act = FLAGS.activation if FLAGS.non_linear_act else None
		self.is_discrete = FLAGS.is_discrete
		self.learn_A = FLAGS.learn_A

		self.activation = FLAGS.activation
		self.optimizer = FLAGS.optimizer
		self.lr = FLAGS.lr   

		# graph ops+variables
		self.x = tf.placeholder(self.datasource.dtype, shape=[None, self.input_height, self.input_width, self.input_channels], name='vae_input')
		self.noise_std = tf.placeholder_with_default(FLAGS.noise_std, shape=(), name='noise_std')
		self.reg_param = tf.placeholder_with_default(FLAGS.reg_param, shape=(), name='reg_param')
		self.mean, self.z, self.x_reconstr_logits = self.create_computation_graph(self.x)
		
		if vae:
			z_mean_logcov = tf.concat([self.mean, tf.log(self.noise_std)*tf.ones_like(self.mean)], axis=-1)
			self.loss, self.reconstr_loss = self.get_vae_loss(self.x, z_mean_logcov, self.x_reconstr_logits)
		else:
			self.loss, self.reconstr_loss = self.get_loss(self.x, self.x_reconstr_logits)

		# session ops
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		if self.learn_A:
			train_vars = tf.trainable_variables()
		else:
			A, A_val = self.get_A()
			self.assign_A_op = tf.assign(A, A_val)
			train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/dec')
		

		self.train_op = self.optimizer(learning_rate=self.lr).minimize(self.loss, 
			global_step=self.global_step, var_list=train_vars)

		# summary ops
		self.summary_op = tf.summary.merge_all()

		# session ops
		self.init_op = tf.global_variables_initializer()
		self.saver = tf.train.Saver(max_to_keep=None)


	def encoder(self, x, reuse=True):
		"""
		more complex encoder architecture for images with more than 1 color channel
		""" 
		# print('hello')
		# print(x.get_shape())
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('encoder', reuse=reuse):
				conv1 = tf.layers.conv2d(x, 32, 4, strides=(2,2), padding="SAME", activation=tf.nn.relu, kernel_regularizer=regularizer, reuse=reuse, name='conv1')
				# print('L74', conv1.get_shape())
				conv2 = tf.layers.conv2d(conv1, 32, 4, strides=(2,2), padding="SAME", activation=tf.nn.relu, kernel_regularizer=regularizer, reuse=reuse, name='conv2')
				# print(conv2.get_shape())
				conv3 = tf.layers.conv2d(conv2, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.relu, kernel_regularizer=regularizer, reuse=reuse, name='conv3')
				# print(conv3.get_shape())
				conv4 = tf.layers.conv2d(conv3, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.relu, kernel_regularizer=regularizer, reuse=reuse, name='conv4')
				# print(conv4.get_shape())
				conv5 = tf.layers.conv2d(conv4, 256, 4, padding="VALID", activation=tf.nn.relu, kernel_regularizer=regularizer, reuse=reuse, name='conv5')
				# print(conv5.get_shape())
				flattened = tf.reshape(conv5, (-1, 256*1*1))
				# print(flattened.get_shape())
				z_mean = tf.layers.dense(flattened, self.z_dim, activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-final')
				# print(z_mean.get_shape())
		return z_mean


	def decoder(self, z, reuse=True):
		"""
		more complex decoder architecture for images with more than 1 color channel (e.g. celebA)
		"""
		z = tf.convert_to_tensor(z)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('decoder', reuse=reuse):
				d = tf.layers.dense(z, 256, activation=tf.nn.relu, use_bias=False, reuse=reuse, name='fc1')		
				d = tf.reshape(d, (-1, 1, 1, 256))
				deconv1 = tf.layers.conv2d_transpose(d, 256, 4, padding="VALID", activation=tf.nn.relu, reuse=reuse, name='deconv1')
				deconv2 = tf.layers.conv2d_transpose(deconv1, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.relu, reuse=reuse, name='deconv2')
				deconv3 = tf.layers.conv2d_transpose(deconv2, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.relu, reuse=reuse, name='deconv3')
				deconv4 = tf.layers.conv2d_transpose(deconv3, 32, 4, strides=(2,2), padding="SAME", activation=tf.nn.relu, reuse=reuse, name='deconv4')
				# output channel = 3; TODO: you may or may not want the sigmoid activation
				output = tf.layers.conv2d_transpose(deconv4, 3, 4, strides=(2,2), padding="SAME", activation=tf.nn.sigmoid, reuse=reuse, name='deconv5')
		return output


	def create_computation_graph(self, x, std=0.1, reuse=False):

		mean = self.encoder(x, reuse=reuse)
		eps = tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)
		z = tf.add(mean, tf.multiply(std, eps))
		x_reconstr_logits = self.decoder(z, reuse=reuse)

		return mean,z, x_reconstr_logits

	def get_vae_loss(self, x, z_mean_logcov, x_reconstr_logits):

		# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		# 		logits=x_reconstr_logits, labels=x))
		if self.is_discrete:
			# assuming Bernoulli observation model for decoder
			loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
				logits=x_reconstr_logits, labels=x))
		else:
			# assuming Gaussian observation model with fixed variance for decoder
			loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[1,2,3]))

		z_dim = self.z_dim
		latent_loss = -0.5 * tf.reduce_sum(1 + z_mean_logcov[:, z_dim:]
						   - tf.square(z_mean_logcov[:, :z_dim]) 
						   - tf.exp(z_mean_logcov[:, z_dim:]), 1)

		tf.summary.scalar('reconstruction loss', loss)

		total_loss = loss + latent_loss

		return total_loss, loss


	def get_loss(self, x, x_reconstr_logits):

		# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		# 		logits=x_reconstr_logits, labels=x))
		if self.is_discrete:
			# assuming Bernoulli observation model for decoder
			loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
				logits=x_reconstr_logits, labels=x))
		else:
			# assuming Gaussian observation model with fixed variance for decoder
			loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[1,2,3]))

		tf.summary.scalar('reconstruction loss', loss)

		return loss, loss

	def train(self, ckpt=None, verbose=True):
		"""
		Trains VAE for specified number of epochs.
		"""
		
		sess = self.sess
		datasource = self.datasource

		if FLAGS.resume:
			if ckpt is None:
				ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
			self.saver.restore(sess, ckpt)
		else:
			sess.run(self.init_op)
			if not self.learn_A:
				sess.run(self.assign_A_op)

		t0 = time.time()
		train_dataset = datasource.get_dataset('train')
		train_dataset = train_dataset.batch(FLAGS.batch_size)
		train_dataset = train_dataset.shuffle(buffer_size=10000)
		train_iterator = train_dataset.make_initializable_iterator()
		next_train_batch = train_iterator.get_next()

		valid_dataset = datasource.get_dataset('valid')
		valid_dataset = valid_dataset.batch(FLAGS.batch_size*10)
		valid_iterator = valid_dataset.make_initializable_iterator()
		next_valid_batch = valid_iterator.get_next()

		epoch_train_losses = []
		epoch_valid_losses = []
		epoch_save_paths = []

		for epoch in range(FLAGS.num_epochs):
			sess.run(train_iterator.initializer)
			sess.run(valid_iterator.initializer)
			epoch_train_loss = 0.
			num_batches = 0.
			while True:
				try:
					x = sess.run(next_train_batch)
					feed_dict = {self.x: x}
					sess.run(self.train_op, feed_dict)
					batch_loss, train_summary, gs = sess.run([self.reconstr_loss, self.summary_op, self.global_step], feed_dict)
					epoch_train_loss += batch_loss
					num_batches += 1
				except tf.errors.OutOfRangeError:
					break
			if verbose:
				epoch_train_loss /= num_batches
				x = sess.run(next_valid_batch)
				epoch_valid_loss, valid_summary, gs = sess.run([self.reconstr_loss, self.summary_op, self.global_step], feed_dict={self.x: x})
				print('Epoch {}, l2 train loss: {:0.6f}, l2 valid loss: {:0.6f}, time: {}s'. \
						format(epoch+1, np.sqrt(epoch_train_loss), np.sqrt(epoch_valid_loss), int(time.time()-t0)))
				sys.stdout.flush()
				save_path = self.saver.save(sess, os.path.join(FLAGS.logdir, 'model.ckpt'), global_step=gs)
				epoch_train_losses.append(epoch_train_loss)
				epoch_valid_losses.append(epoch_valid_loss)
				epoch_save_paths.append(save_path)

		best_ckpt = None
		if verbose:
			min_idx = epoch_valid_losses.index(min(epoch_valid_losses))
			print('Restoring ckpt at epoch', min_idx+1,'with lowest validation error:', epoch_save_paths[min_idx])
			best_ckpt = epoch_save_paths[min_idx]

		return (epoch_train_losses, epoch_valid_losses), best_ckpt

	def test(self, ckpt=None):

		sess = self.sess
		datasource = self.datasource

		if ckpt is None:
			ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
		self.saver.restore(sess, ckpt)

		test_dataset = datasource.get_dataset('test')
		test_dataset = test_dataset.batch(FLAGS.batch_size*10)
		test_iterator = test_dataset.make_initializable_iterator()
		next_test_batch = test_iterator.get_next()

		test_loss = 0.
		num_batches = 0.
		sess.run(test_iterator.initializer)
		while True:
			try:
				x = sess.run(next_test_batch)
				batch_test_loss =sess.run(self.reconstr_loss, feed_dict={self.x: x})
				test_loss += batch_test_loss
				# print(np.sqrt(batch_test_loss))
				x_reconstr_logits = sess.run(self.x_reconstr_logits, feed_dict={self.x: x})
				num_batches += 1.
			except tf.errors.OutOfRangeError:
				break
		test_loss /= num_batches
		print('L2 squared test loss (per image): {:0.6f}'.format(test_loss))
		print('L2 squared test loss (per pixel): {:0.6f}'.format(test_loss/self.input_dim))

		print('L2 test loss (per image): {:0.6f}'.format(np.sqrt(test_loss)))
		print('L2 test loss (per pixel): {:0.6f}'.format(np.sqrt(test_loss)/self.input_dim))

		return test_loss

	def reconstruct(self, ckpt=None, pkl_file=None):

		import pickle

		sess = self.sess
		datasource = self.datasource

		if ckpt is None:
			ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
		self.saver.restore(sess, ckpt)

		if pkl_file is None:
			test_dataset = datasource.get_dataset('test')
			test_dataset = test_dataset.batch(10)
			test_iterator = test_dataset.make_initializable_iterator()
			next_test_batch = test_iterator.get_next()

			sess.run(test_iterator.initializer)
			x = sess.run(next_test_batch)
		else:
			with open(pkl_file, 'rb') as f:
				images = pickle.load(f)
			x = np.vstack([images[i] for i in range(10)])
		print(x.shape)
		print(np.max(x), np.min(x))
		with open(os.path.join(FLAGS.outdir, 'original.pkl'), 'wb') as f:
			pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
		x_reconstr_logits = sess.run(self.x_reconstr_logits, feed_dict={self.x: x})
		print(np.max(x_reconstr_logits), np.min(x_reconstr_logits))
		plot_colored(np.vstack((x, x_reconstr_logits)), m=10, n=2, title='reconstructions')
		
		with open(os.path.join(FLAGS.outdir, 'reconstr.pkl'), 'wb') as f:
			pickle.dump(x_reconstr_logits, f, pickle.HIGHEST_PROTOCOL)
		return x_reconstr_logits


def conv_cond_concat(x, y):
	"""Concatenate conditioning vector on feature map axis."""
	x_shapes = x.get_shape()
	y_shapes = y.get_shape()
	return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, 
		k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
		name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
				  initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

	return conv

def deconv2d(input_, output_shape,
	   k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
	   name="deconv2d", with_w=False):
	with tf.variable_scope(name):
	# filter : [height, width, output_channels, in_channels]
		w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
			  initializer=tf.random_normal_initializer(stddev=stddev))

	try:
		deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
				strides=[1, d_h, d_w, 1])
	# Support for verisons of TensorFlow before 0.7.0
	except AttributeError:
		deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
				strides=[1, d_h, d_w, 1])

	biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
	deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

	if with_w:
		return deconv, w, biases
	else:
		return deconv
	 
def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
					 tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
		  initializer=tf.constant_initializer(bias_start))

	if with_w:
		return tf.matmul(input_, matrix) + bias, matrix, bias
	else:
		return tf.matmul(input_, matrix) + bias