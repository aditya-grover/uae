# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(dataset, name):
  """Converts a dataset to tfrecords."""
  images = dataset.images
  labels = dataset.labels
  num_examples = labels.shape[0]

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))

  filename = os.path.join(FLAGS.directory, FLAGS.dataset, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(int(labels[index])),
        'features': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


def main(unused_argv):
  # Get the data.

  if FLAGS.dataset == 'mnist':
    datasets = mnist.read_data_sets(FLAGS.directory,
                                     dtype=tf.uint8,
                                     reshape=False,
                                     validation_size=FLAGS.valid_size)

    # Convert to Examples and write the result to TFRecords.
    convert_to(datasets.train, 'mnist_train')
    convert_to(datasets.validation, 'mnist_valid')
    convert_to(datasets.test, 'mnist_test')
  elif FLAGS.dataset == 'omniglot':
    import h5py
    file_path = os.path.join(FLAGS.directory, FLAGS.dataset, 'omniglot.hdf5')
    if not os.path.exists(file_path):
      raise ValueError('need: ', file_path)
    f = h5py.File(file_path, 'r')

    trainimages = f['train']
    validimages = f['valid']
    testimages = f['test']

    trainlabels = f['trainlabels']
    validlabels = f['validlabels']
    testlabels = f['testlabels']

    trainlabels2 = f['trainlabels2']
    validlabels2 = f['validlabels2']
    testlabels2 = f['testlabels2']

    print(trainimages.shape, validimages.shape, testimages.shape)
    print(trainlabels.shape, validlabels.shape, testlabels.shape)
    print(trainlabels2.shape, validlabels2.shape, testlabels2.shape)

    from types import SimpleNamespace
    train_dataset = SimpleNamespace(images= trainimages, labels= trainlabels, labels2= trainlabels2)
    valid_dataset = SimpleNamespace(images= validimages, labels= validlabels, labels2= validlabels2)
    test_dataset = SimpleNamespace(images= testimages, labels= testlabels, labels2= testlabels2)

    # {'images': trainimages, 'labels': trainlabels, 'labels2': trainlabels2}
    # valid_dataset = {'images': validimages, 'labels': validlabels, 'labels2': validlabels2}
    # test_dataset = {'images': testimages, 'labels': testlabels, 'labels2': testlabels2}

    convert_to(train_dataset, 'omniglot_train')
    convert_to(valid_dataset, 'omniglot_valid')
    convert_to(test_dataset, 'omniglot_test')
  else:
    raise NotImplementedError


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset',
      type=str,
      default='omniglot',
      help='Dataset (mnist/omniglot)'
  )
  parser.add_argument(
      '--directory',
      type=str,
      default='../datasets',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--valid_size',
      type=int,
      default=10000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
