import numpy as np
from scipy.special import expit
import inspect
import importlib
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

def load_dynamic(class_name, module_name):
	"""
	Load a class dynamically from the classname.
	:param class_name: string
	"""
	return getattr(importlib.import_module(module_name), class_name)


def get_arglist(func):
	"""
	Get the argument list of a function.
	:param func: function handle
	:return arglist: list of argument names
	"""
	argspec = inspect.getfullargspec(func)
	return argspec[0]


def get_args(arglist, config):
	"""
	Get the argument values for arguments in an argument list
	from a configuration dictionary.
	:param arglist: list of strings
	:param conf: configuration dict
	"""
	args = {}
	for argname in arglist:
		# do our best and hope that failures mean default values are present
		try:
			args[argname] = config[argname]
		except:
			pass
	return args


def sigmoid(x, gamma=1):
	"""
	Sigmoid function (numerically stable).
	"""
	u = gamma * x
	return expit(u)

def provide_unlabelled_data(data, batch_size=10):
	"""
	Provide batches of data; data = X
	"""
	N = len(data)
	# Create indices for data
	X_indexed = list(zip(range(N), np.split(data, N)))
	def data_iterator():
		while True:
			idxs = np.arange(0, N)
			np.random.shuffle(idxs)
			X_shuf = [X_indexed[idx] for idx in idxs]
			for batch_idx in range(0, N, batch_size):
				X_shuf_batch = X_shuf[batch_idx:batch_idx+batch_size]
				indices, X_batch = zip(*X_shuf_batch)
				X_batch = np.vstack(X_batch)
				yield indices, X_batch
	return data_iterator()


def provide_data(data, batch_size=10):
	"""
	Provide batches of data; data = (X, y).
	"""
	N = len(data[0])
	X_mean = np.mean(data[0], axis=0)
	X_std = np.std(data[0], axis=0)
	y_mean = np.mean(data[1], axis=0)
	y_std = np.std(data[1], axis=0)
	# Create indices for data
	X_indexed = list(zip(range(N), np.split(data[0], N)))
	y_indexed = list(zip(range(N), np.split(data[1], N)))
	def data_iterator():
		while True:
			idxs = np.arange(0, N)
			np.random.shuffle(idxs)
			X_shuf = [X_indexed[idx] for idx in idxs]
			y_shuf = [y_indexed[idx] for idx in idxs]
			for batch_idx in range(0, N, batch_size):
				X_shuf_batch = X_shuf[batch_idx:batch_idx+batch_size]
				y_shuf_batch = y_shuf[batch_idx:batch_idx+batch_size]
				indices, X_batch = zip(*X_shuf_batch)
				_, y_batch = zip(*y_shuf_batch)
				X_batch = np.vstack(X_batch)
				y_batch = np.vstack(y_batch)
				yield indices, X_batch, y_batch
	return data_iterator(), X_mean, X_std, y_mean, y_std


# def fc(x, weights_shape, bias_shape, 
#     weights_initializer=tf.contrib.layers.xavier_initializer(),
#     bias_initializer=tf.constant_initializer()):
#     """
#     Creates a fully connected neural net layer.
#     """
#     weights = tf.get_variable('weights',  weights_shape, 
#        initializer=weights_initializer)
#     biases = tf.get_variable('biases', bias_shape,
#         initializer=bias_initializer)
#     return tf.nn.xw_plus_b(x, weights, biases)


def plot(samples, m=4, n=None, px=28, title=None):
	"""
	Plots samples.
		n: Number of rows and columns; n^2 samples
		px: Pixels per side for each sample
	"""
	if n is None:
		n = m
	fig = plt.figure(figsize=(m, n))
	gs = gridspec.GridSpec(n, m)
	gs.update(wspace=0.05, hspace=0.05)
	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(px, px), cmap='Greys')
	if title is None:
		title = 'samples'
	fig.savefig(os.path.join(FLAGS.outdir, title))
	# fig.show()
	plt.close()
	return fig


def plot_colored(samples, m=4, n=None, px=64, title=None):
	"""
	Plots samples.
		n: Number of rows and columns; n^2 samples
		px: Pixels per side for each sample
	"""
	import skimage.io as io

	if n is None:
		n = m
	fig = plt.figure(figsize=(m, n))
	gs = gridspec.GridSpec(n, m)
	gs.update(wspace=0.05, hspace=0.05)
	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		# plt.imshow(sample, cmap='Greys')
		io.imshow(sample)
		io.show()
	if title is None:
		title = 'samples'
	fig.savefig(os.path.join(FLAGS.outdir, title))
	# fig.show()
	plt.close()
	return fig


def get_activation_fn(activation):
	"""
	Returns the specified tensorflow activation function.
	"""
	if activation == 'tanh':
		return tf.tanh
	elif activation == 'sigmoid':
		return tf.sigmoid
	elif activation == 'softplus':
		return tf.nn.softplus
	else:
		return tf.nn.relu # default


def get_optimizer_fn(optimizer):
	"""
	Returns the specified tensorflow optimizer.
	"""
	if optimizer == 'sgd':
		return tf.train.GradientDescentOptimizer
	elif optimizer == 'momentum':
		return tf.train.MomentumOptimizer
	elif optimizer == 'rmsprop':
		return tf.train.RMSPropOptimizer
	else:
		return tf.train.AdamOptimizer # default



