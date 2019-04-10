

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
np.random.seed(0)

import tensorflow as tf
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

import sys

flags.DEFINE_string('datadir', '../datasets/', 'directory for datasets')
flags.DEFINE_string('outdir', './results/', 'directory to save samples, final results')
flags.DEFINE_string('datasource', 'mnist', 'mnist/omniglot')
flags.DEFINE_integer('num_measurements', 10, 'number of measurements')
flags.DEFINE_string('baseline_file', 'pca', 'pca/uae file')
flags.DEFINE_string('method', 'pca', 'pca/uae')
flags.DEFINE_bool('dump', True, 'Dumps to log.txt if True')

def run_classifier(f_Xtrain, Ytrain, f_Xtest, Ytest):

	from sklearn.neural_network import MLPClassifier
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.svm import SVC
	from sklearn.gaussian_process import GaussianProcessClassifier
	from sklearn.gaussian_process.kernels import RBF
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
	from sklearn.naive_bayes import GaussianNB
	from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

	names = ["Nearest Neighbors", 
		 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
		 "Naive Bayes", "QDA", "Linear SVM"
		 # , "RBF SVM"
		 ]

	classifiers = [
		KNeighborsClassifier(),
		DecisionTreeClassifier(),
		RandomForestClassifier(),
		MLPClassifier(),
		AdaBoostClassifier(),
		GaussianNB(),
		QuadraticDiscriminantAnalysis(),
		SVC(kernel="linear")
		# ,SVC(gamma=2, C=1)
		]


	for name, clf in zip(names, classifiers):
		clf.fit(f_Xtrain, Ytrain)
		score = clf.score(f_Xtest, Ytest)
		print(name, score)
		sys.stdout.flush()




def get_projection_matrix():

	A = np.load(FLAGS.baseline_file)
	assert (A.shape[1] == FLAGS.num_measurements)
	return A


def get_data():

	if FLAGS.datasource == 'mnist':
		from tensorflow.examples.tutorials.mnist import input_data
		mnist = input_data.read_data_sets(os.path.join(FLAGS.datadir, FLAGS.datasource), one_hot=False)
		trainX = np.vstack((mnist.train.images, mnist.validation.images))
		trainY = np.squeeze(np.vstack((mnist.train.labels.reshape((-1,1)), mnist.validation.labels.reshape((-1,1)))))

		testX = mnist.test.images
		testY = np.squeeze(mnist.test.labels.reshape((-1,1)))
	elif FLAGS.datasource == 'omniglot':
		import h5py
		file_path = os.path.join(FLAGS.datadir, FLAGS.datasource, 'omniglot.hdf5')
		if not os.path.exists(file_path):
		  raise ValueError('need: ', file_path)
		f = h5py.File(file_path, 'r')
		trainX = np.vstack((f['train'], f['valid']))
		trainY = np.squeeze(np.vstack((f['trainlabels'], f['validlabels'])))
		
		testX = f['test'][:]
		testY = np.squeeze(f['testlabels'][:])
	else:
		raise NotImplementedError

	return trainX, trainY, testX, testY



def main():

	FLAGS.outdir = os.path.join(FLAGS.outdir, FLAGS.datasource, FLAGS.method, str(FLAGS.num_measurements))
	if not os.path.exists(FLAGS.outdir):
		os.makedirs(FLAGS.outdir)

	import json
	with open(os.path.join(FLAGS.outdir, 'config.json'), 'w') as fp:
	    json.dump(tf.app.flags.FLAGS.flag_values_dict(), fp, indent=4, separators=(',', ': '))

	if FLAGS.dump:
		sys.stdout = open(os.path.join(FLAGS.outdir, 'log.txt'), 'w')

	trainX, trainY, testX, testY = get_data()
	print(trainX.shape, trainY.shape, testX.shape, testY.shape)
	A = get_projection_matrix()

	f_trainX = trainX.dot(A)
	f_testX = testX.dot(A)
	
	run_classifier(f_trainX, trainY, f_testX, testY)


if __name__ == '__main__':

	main()
