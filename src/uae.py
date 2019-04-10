from utils import *
import tensorflow as tf 
import numpy as np
import time
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

class UAE():
	# uncertainty autoencoder

	def __init__(self, sess, datasource, vae=False):

		self.seed = FLAGS.seed
		tf.set_random_seed(self.seed)
		np.random.seed(self.seed)

		self.sess = sess
		self.datasource = datasource
		self.input_dim = self.datasource.input_dim
		self.z_dim = FLAGS.num_measurements
		self.dec_layers = [self.input_dim] + FLAGS.dec_arch
		self.enc_layers = FLAGS.enc_arch + [self.z_dim]
		self.transfer = FLAGS.transfer

		self.learn_A = FLAGS.learn_A
		self.activation = FLAGS.activation
		self.optimizer = FLAGS.optimizer
		self.lr = FLAGS.lr   

		# graph ops+variables
		self.x = tf.placeholder(self.datasource.dtype, shape=[None, self.input_dim], name='vae_input')
		self.noise_std = tf.placeholder_with_default(FLAGS.noise_std, shape=(), name='noise_std')
		self.reg_param = tf.placeholder_with_default(FLAGS.reg_param, shape=(), name='reg_param')
		self.mean, self.z, self.x_reconstr_logits = self.create_computation_graph(self.x, std=self.noise_std)

		if vae:
			z_mean_logcov = tf.concat([self.mean, tf.log(self.noise_std)*tf.ones_like(self.mean)], axis=-1)
			self.loss, self.reconstr_loss = self.get_vae_loss(self.x, z_mean_logcov, self.x_reconstr_logits)
		else:
			self.loss, self.reconstr_loss = self.get_loss(self.x, self.x_reconstr_logits)

		# session ops
		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		if self.transfer == 1:
			train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/dec')
		elif self.transfer == 2:
			train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/enc')
		elif self.learn_A:
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
		Specifies the parameters for the mean and variance of p(y|x)
		"""

		e = x
		enc_layers = self.enc_layers
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('encoder', reuse=reuse):
				for layer_idx, layer_dim in enumerate(enc_layers[:-1]):
					e = tf.layers.dense(e, layer_dim, activation=self.activation, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(layer_idx))
				z_mean = tf.layers.dense(e, enc_layers[-1], activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(len(enc_layers)))
		
		return z_mean


	def decoder(self, z, reuse=True, use_bias=False):

		d = tf.convert_to_tensor(z)
		dec_layers = self.dec_layers

		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('decoder', reuse=reuse):
				for layer_idx, layer_dim in list(reversed(list(enumerate(dec_layers))))[:-1]:
					d = tf.layers.dense(d, layer_dim, activation=self.activation, reuse=reuse, name='fc-' + str(layer_idx), use_bias=use_bias)
				x_reconstr_logits = tf.layers.dense(d, dec_layers[0], activation=tf.nn.sigmoid, reuse=reuse, name='fc-0', use_bias=use_bias) # clip values between 0 and 1

		return x_reconstr_logits


	def create_computation_graph(self, x, std=0.1, reuse=False):

		mean = self.encoder(x, reuse=reuse)
		eps = tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)
		z = tf.add(mean, tf.multiply(std, eps))
		x_reconstr_logits = self.decoder(z, reuse=reuse)

		return mean, z, x_reconstr_logits


	def get_vae_loss(self, x, z_mean_logcov, x_reconstr_logits):

		reg_loss = tf.losses.get_regularization_loss() 
		reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=1))

		z_dim = self.z_dim
		latent_loss = -0.5 * tf.reduce_sum(1 + z_mean_logcov[:, z_dim:]
						   - tf.square(z_mean_logcov[:, :z_dim]) 
						   - tf.exp(z_mean_logcov[:, z_dim:]), 1)

		tf.summary.scalar('reconstruction loss', reconstr_loss)

		total_loss = reconstr_loss + reg_loss + latent_loss

		return total_loss, reconstr_loss


	def get_loss(self, x, x_reconstr_logits):

		reg_loss = tf.losses.get_regularization_loss() 
		reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=1))

		tf.summary.scalar('reconstruction loss', reconstr_loss)

		total_loss = reconstr_loss + reg_loss

		return total_loss, reconstr_loss

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
		elif self.transfer > 0:
			log_file = os.path.join(FLAGS.transfer_outdir, 'log.txt')
			if os.path.exists(log_file):
				for line in open(log_file):
					if "Restoring ckpt at epoch" in line:
						ckpt = line.split()[-1]
						break
			if ckpt is None:
				ckpt = tf.train.latest_checkpoint(FLAGS.transfer_logdir)
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
					x = sess.run(next_train_batch)[0]
					feed_dict = {self.x: x}
					sess.run(self.train_op, feed_dict)
					batch_loss, gs = sess.run([self.reconstr_loss, self.global_step], feed_dict)
					epoch_train_loss += batch_loss
					num_batches += 1
				except tf.errors.OutOfRangeError:
					break
			if verbose:
				epoch_train_loss /= num_batches
				x = sess.run(next_valid_batch)[0]
				epoch_valid_loss, gs = sess.run([self.reconstr_loss, self.global_step], feed_dict={self.x: x})
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
				x = sess.run(next_test_batch)[0]
				batch_test_loss = sess.run(self.reconstr_loss, feed_dict={self.x: x})
				test_loss += batch_test_loss
				
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
			x = sess.run(next_test_batch)[0]
		else:
			with open(pkl_file, 'rb') as f:
				images = pickle.load(f)
			x = np.vstack([images[i] for i in range(10)])

		x_reconstr_logits = sess.run(self.x_reconstr_logits, feed_dict={self.x: x})
		print(np.max(x_reconstr_logits), np.min(x_reconstr_logits))
		print(np.max(x), np.min(x))
		plot(np.vstack((x, x_reconstr_logits)), m=10, n=2, title='reconstructions')
		
		with open(os.path.join(FLAGS.outdir, 'reconstr.pkl'), 'wb') as f:
			pickle.dump(x_reconstr_logits, f, pickle.HIGHEST_PROTOCOL)
		return x_reconstr_logits
