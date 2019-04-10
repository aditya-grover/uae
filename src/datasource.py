
""" Code for loading data. """
import numpy as np
import random
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

class Datasource(object):

	def __init__(self, sess):

		self.sess = sess
		self.seed = FLAGS.seed
		tf.set_random_seed(self.seed)
		np.random.seed(self.seed)

		self.batch_size = FLAGS.batch_size

		if FLAGS.datasource == 'mnist' or FLAGS.datasource == 'omniglot2mnist':

			self.target_dataset = 'mnist'
			self.TRAIN_FILE = 'mnist_train.tfrecords'
			self.VALID_FILE = 'mnist_valid.tfrecords'
			self.TEST_FILE = 'mnist_test.tfrecords'

			self.input_dim = 784
			self.num_classes = 10
			self.dtype = tf.float32
			self.preprocess = self._preprocess_mnist
			self.get_dataset = self.get_tf_dataset

		elif FLAGS.datasource == 'omniglot' or FLAGS.datasource == 'mnist2omniglot':

			self.target_dataset = 'omniglot'
			self.TRAIN_FILE = 'omniglot_train.tfrecords'
			self.VALID_FILE = 'omniglot_valid.tfrecords'
			self.TEST_FILE = 'omniglot_test.tfrecords'

			self.input_dim = 784
			self.num_classes = 50
			self.dtype = tf.float32
			self.preprocess = self._preprocess_omniglot
			self.get_dataset = self.get_tf_dataset

		elif FLAGS.datasource == 'celeba':

			self.target_dataset = 'celeba'
			self.TRAIN_FILE = 'celeba_train.tfrecords'
			self.VALID_FILE = 'celeba_valid.tfrecords'
			self.TEST_FILE = 'celeba_test.tfrecords'

			self.input_dim = 64 * 64 * 3
			self.input_height = 64
			self.input_width = 64
			self.input_channels = 3
			self.dtype = tf.float32
			self.preprocess = self._preprocess_celebA
			self.get_dataset = self.get_binary_tf_dataset
		

		else:
			raise NotImplementedError

		train_dataset = self.get_dataset('train')

		return

	def _preprocess_omniglot(self, parsed_example):

		image = tf.decode_raw(parsed_example['features'], tf.float32)
		image.set_shape([self.input_dim])
		label = tf.cast(parsed_example['label'], tf.int32)

		return image, label

	def _preprocess_mnist(self, parsed_example):

		image = tf.decode_raw(parsed_example['features'], tf.uint8)
		image.set_shape([self.input_dim])
		image = tf.cast(image, tf.float32) * (1. / 255)
		label = tf.cast(parsed_example['label'], tf.int32)

		return image, label

	def _preprocess_celebA(self, parsed_example):

		image = tf.decode_raw(parsed_example['features'], tf.uint8)
		image = tf.reshape(image, (self.input_height, self.input_width, self.input_channels))
		# image.set_shape([self.input_dim])
		# convert from bytes to data
		image = tf.divide(tf.to_float(image), 127.5) - 1.0
		# convert back to [0, 1] pixels
		image = tf.clip_by_value(tf.divide(image + 1., 2.), 0., 1.)
		return image		

	def get_tf_dataset_celeba(self, split):

		def _parse_function(example_proto):
			example = {'image_raw': tf.FixedLenFeature((), tf.string, default_value=''),
						'height': tf.FixedLenFeature((), tf.int64, default_value=218),
						'width': tf.FixedLenFeature((), tf.int64, default_value=178),
						'channels': tf.FixedLenFeature((), tf.int64, default_value=3)}
			parsed_example = tf.parse_single_example(example_proto, example)
			preprocessed_features = self.preprocess(parsed_example)
			return preprocessed_features

		filename = os.path.join(FLAGS.datadir, self.target_dataset, self.TRAIN_FILE if split=='train' 
			else self.VALID_FILE if split=='valid' else self.TEST_FILE)
		tf_dataset = tf.data.TFRecordDataset(filename)
		return tf_dataset.map(_parse_function)

	def get_tf_dataset(self, split):

		def _parse_function(example_proto):
			example = {'features': tf.FixedLenFeature((), tf.string, default_value=''),
						  'label': tf.FixedLenFeature((), tf.int64, default_value=0)}
			parsed_example = tf.parse_single_example(example_proto, example)
			preprocessed_features, preprocessed_label = self.preprocess(parsed_example)
			return preprocessed_features, preprocessed_label

		filename = os.path.join(FLAGS.datadir, self.target_dataset, self.TRAIN_FILE if split=='train' 
			else self.VALID_FILE if split=='valid' else self.TEST_FILE)
		tf_dataset = tf.data.TFRecordDataset(filename)
		return tf_dataset.map(_parse_function)

	def get_binary_tf_dataset(self, split):

		def _parse_function(example_proto):
			# no labels available for binary MNIST
			example = {'features': tf.FixedLenFeature((), tf.string, default_value='')}
			parsed_example = tf.parse_single_example(example_proto, example)
			preprocessed_features = self.preprocess(parsed_example)
			return preprocessed_features

		filename = os.path.join(FLAGS.datadir, self.target_dataset, self.TRAIN_FILE if split=='train' 
			else self.VALID_FILE if split=='valid' else self.TEST_FILE)
		tf_dataset = tf.data.TFRecordDataset(filename)
		return tf_dataset.map(_parse_function)


