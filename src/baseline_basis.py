import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('datadir', '../datasets/', 'directory for datasets')
flags.DEFINE_string('outdir', './results/', 'directory to save samples, final results')
flags.DEFINE_string('datasource', 'mnist', 'mnist/omniglot')
flags.DEFINE_integer('num_measurements', 10, 'number of measurements')
flags.DEFINE_string('baseline', 'pca', 'pca/ica/rp')

def get_A_pca(X, num_measurements):
 
	from sklearn.decomposition import PCA
	pca = PCA(n_components=num_measurements)
	pca.fit(X)
	A = pca.components_.T

	return A

def get_A_ica(X, num_measurements):

	from sklearn.decomposition import FastICA
	ica = FastICA(n_components=num_measurements)
	ica.fit(X)
	A = ica.components_.T

	return A

def get_A_rp(X, num_measurements):

	A = np.random.randn(X.shape[1], num_measurements)

	return A

def main():

	if FLAGS.baseline == 'pca':
		get_A = get_A_pca
	elif FLAGS.baseline == 'ica':
		get_A = get_A_ica
	elif FLAGS.baseline == 'rp':
		get_A = get_A_rp
	else:
		raise NotImplementedError

	if FLAGS.datasource == 'mnist':
		from tensorflow.examples.tutorials.mnist import input_data
		mnist = input_data.read_data_sets(os.path.join(FLAGS.datadir, FLAGS.datasource), one_hot=True)
		X = mnist.train.images
	elif FLAGS.datasource == 'omniglot':
		import h5py
		file_path = os.path.join(FLAGS.datadir, FLAGS.datasource, 'omniglot.hdf5')
		if not os.path.exists(file_path):
		  raise ValueError('need: ', file_path)
		f = h5py.File(file_path, 'r')
		X = f['train']
		print(X.shape)
	else:
		raise NotImplementedError
	
	file = os.path.join(FLAGS.outdir, FLAGS.datasource, FLAGS.baseline + '_' + str(FLAGS.num_measurements) + '.npy')
	if os.path.exists(file):
		A = np.load(file)
	else:
		A = get_A(X, FLAGS.num_measurements)
	np.save(file, A)


if __name__ == "__main__":
	main()

