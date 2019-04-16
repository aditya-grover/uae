import numpy as np 
import tensorflow as tf 
from utils import *
from datasource import Datasource

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

# File options
flags.DEFINE_string('datadir', './datasets/', 'directory for datasets')
flags.DEFINE_string('datasource', 'mnist', 'mnist/omniglot/mnist2omniglot/omniglot2mnist')
flags.DEFINE_string('logdir', './models/', 'directory to save checkpoints, events files')
flags.DEFINE_string('outdir', './results/', 'directory to save samples, final results')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train.')
flags.DEFINE_bool('test', True, 'True to test.')
flags.DEFINE_string('ckpt', None, 'ckpt to load if resume is True. Defaults (None) to latest ckpt in logdir')
flags.DEFINE_string('exp_id', '0', 'exp_id appended to logdir and outdir')
flags.DEFINE_string('gpu_id', '0', 'gpu id options')
flags.DEFINE_bool('dump', True, 'Dumps to log.txt if True')

# Training options
flags.DEFINE_integer('num_epochs', 500, 'number of training epochs')
flags.DEFINE_integer('batch_size', 100, 'number of datapoints per batch')
flags.DEFINE_float('lr', 0.001, 'learning rate for the model')
flags.DEFINE_string('optimizer', 'adam', 'sgd, adam, momentum')
flags.DEFINE_integer('log_interval', 500, 'training steps after which summary and checkpoints dumped')
flags.DEFINE_integer('num_samples', 16, 'number of samples to generate')
flags.DEFINE_bool('vae', False, 'uses vae loss if True')

# Model options
flags.DEFINE_string('model', 'uae', 'uae/convuae')
flags.DEFINE_string('activation', 'relu', 'sigmoid/tanh/softplus/relu')
flags.DEFINE_integer('seed', 0, 'random seed for initializing model parameters')
flags.DEFINE_string('dec_arch', '500,500', 'comma-separated decoder architecture')
flags.DEFINE_string('enc_arch', '', 'comma-separated encoder architecture')
flags.DEFINE_integer('num_measurements', 10, 'number of measurements')
flags.DEFINE_float('noise_std', 0, 'std. of noise')
flags.DEFINE_float('reg_param', 0., 'regularization for encoder')
flags.DEFINE_bool('learn_A', True, 'learns the measurement matrix if True')
flags.DEFINE_string('A_file', './results/0/A.npy', 'file for precomputed measurement matrix')
flags.DEFINE_bool('non_linear_act', False, 'nonlinear activation on final layer of encoder if True')

flags.DEFINE_integer('transfer', 0, '0: no transfer, 1: transfer encoder 2: transfer decoder')
flags.DEFINE_string('transfer_outdir', './results/', 'directory containing log.txt for source domain')
flags.DEFINE_string('transfer_logdir', './models/', 'directory containing ckpts for source domain')
flags.DEFINE_string('pkl_file', None, 'pkl file for reconstruction')


def process_flags():
	"""
	Processes easy-to-specify cmd line FLAGS to appropriate syntax
	"""
	FLAGS.optimizer = get_optimizer_fn(FLAGS.optimizer)
	FLAGS.activation = get_activation_fn(FLAGS.activation)
	
	if FLAGS.dec_arch == '':
		FLAGS.dec_arch = []
	else:
		FLAGS.dec_arch = list(map(int, FLAGS.dec_arch.split(',')))
	
	if FLAGS.enc_arch == '':
		FLAGS.enc_arch = []
	else:
		FLAGS.enc_arch = list(map(int, FLAGS.enc_arch.split(',')))

	return

def main():
	"""
	Runs the ML loop. Preprocesses data, trains model, along with regular validation and testing.
	"""	
	
	os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)
	subpath = 'noise_' + str(FLAGS.noise_std) 
	FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.datasource, subpath, FLAGS.exp_id)
	FLAGS.outdir = os.path.join(FLAGS.outdir, FLAGS.datasource, subpath, FLAGS.exp_id)

	if FLAGS.transfer > 0:
		if FLAGS.transfer == 2:
			transfer_exp_id = str(int(int(FLAGS.exp_id)/10))
		source = FLAGS.datasource.split("2")[0]
		FLAGS.transfer_logdir = os.path.join(FLAGS.transfer_logdir, source, subpath, transfer_exp_id)
		FLAGS.transfer_outdir = os.path.join(FLAGS.transfer_outdir, source, subpath, transfer_exp_id)
	
	if not os.path.exists(FLAGS.logdir):
		os.makedirs(FLAGS.logdir)
	if not os.path.exists(FLAGS.outdir):
		os.makedirs(FLAGS.outdir)

	import json
	with open(os.path.join(FLAGS.outdir, 'config.json'), 'w') as fp:
	    json.dump(tf.app.flags.FLAGS.flag_values_dict(), fp, indent=4, separators=(',', ': '))

	if FLAGS.dump:
		import sys
		sys.stdout = open(os.path.join(FLAGS.outdir, 'log.txt'), 'w')

	process_flags()
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
	
	datasource = Datasource(sess)
	model_class = load_dynamic(FLAGS.model.upper(), FLAGS.model)
	model = model_class(sess, datasource, vae=FLAGS.vae)

	# run computational graph
	best_ckpt = None
	if FLAGS.train:
		learning_curves, best_ckpt = model.train()
	
	if FLAGS.test:
		if best_ckpt is None:
			log_file = os.path.join(FLAGS.outdir, 'log.txt')
			if os.path.exists(log_file):
				for line in open(log_file):
					if "Restoring ckpt at epoch" in line:
						best_ckpt = line.split()[-1]
						break
		model.test(ckpt=best_ckpt)
		model.reconstruct(ckpt=best_ckpt, pkl_file=FLAGS.pkl_file)

if __name__ == "__main__":
	main()

