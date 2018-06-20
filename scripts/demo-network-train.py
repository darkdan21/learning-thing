#!/bin/usr/env python3

# Based on: https://www.tensorflow.org/tutorials/layers

# Core modules
import argparse
import gzip
import logging
import os.path

# 3rd party modules
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

tf.logging.set_verbosity(tf.logging.INFO)


def load_mnist(directory):
    """
    Load the mnist dataset from files in the directory.

    Based on: https://schwalbe10.github.io/thinkage/2017/03/05/mnist.html

    Returns a dict with keys:
      * train: [numpy array of training images]
      * trnlabl: [numpy array of labels]
      * test: [numpy array of test images]
      * tstlabl: [numpy array of test labels]

    Files are expected to be names as per the MNIST website:
      * train-images-idx3-ubyte.gz
      * train-labels-idx1-ubyte.gz
      * t10k-images-idx3-ubyte.gz
      * t10k-labels-idx1-ubyte.gz
    """

    result = {}

    # Load images
    for (filename, key) in (('train-images-idx3-ubyte.gz', 'train'),
        ('t10k-images-idx3-ubyte.gz', 'test')):
        file_ = os.path.join(directory, filename)
        logger.debug('Loading images from %s', file_)
        with gzip.open(file_, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        result[key] = data.reshape(-1, 28*28)

    # Load labels
    for (filename, key) in (('train-labels-idx1-ubyte.gz', 'trnlabl'),
        ('t10k-labels-idx1-ubyte.gz', 'tstlabl')):
        file_ = os.path.join(directory, filename)
        logger.debug('Loading labels from %s', file_)
        with gzip.open(file_, 'rb') as f:
            result[key] = np.frombuffer(f.read(), np.uint8, offset=8)

    return result

def cnn_model_fn(features, labels, mode):
  """Our actual Nerual Network model"""
  # Input Layer
  input_layer = tf.reshape(tf.cast(features["image"], tf.float32), [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,export_outputs={
            'classify': tf.estimator.export.PredictOutput(tf.argmax(input=logits, axis=1))
        })

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int32), logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    #OPTIMISERS
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    #optimizer = tf.train.AdagradOptimizer
    #optimizer = tf.train.AdagradDAOptimizer
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def build_nn(model_dir):
    """
    Create and setup the neural network.

    model_dir: Temporary directory for model to checkpoint to.
    """
    logger.debug("Setting up neural network")
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)
    return mnist_classifier


def train_nn(network, mnist_data, epochs):
    """
    Train the model.

    network: Neural network to train.
    mnist_data: mnist dataset loaded by load_mnist
    epochs: how many epochs to train for
    """
    logger.debug("Training neural network")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"image": mnist_data['train']},
        y=mnist_data['trnlabl'],
        batch_size=100,
        num_epochs=epochs,
        shuffle=True)
    network.train(input_fn=train_input_fn)

def validate_nn(network, mnist_data):
    """
    Validate the model.

    network: Trained neural network
    mnist_data: mnist dataset loaded by load_mnist
    """
    logger.debug("Validate neural network")

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"image": mnist_data['test']},
        y=mnist_data['tstlabl'],
        num_epochs=1,
        shuffle=False)
    eval_results = network.evaluate(input_fn=eval_input_fn)
    logger.info("Validation results: %s", eval_results)

def save_nn(network, directory):
    """
    Save the network to the directory, ready to be used for inference.

    network: Trained network to save
    directory: directory to save to
    """
    image = tf.placeholder(tf.float32, [None, 28, 28])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': image,
    })
    network.export_savedmodel(directory, input_fn)

if __name__ == "__main__":
    # Parse arguments
    argparser = argparse.ArgumentParser(
        description="MNIST test image pre-processor")

    # Logging options
    output_group=argparser.add_argument_group("output options")
    output_group.add_argument('-l', '--level',
        choices=['debug', 'info', 'warn', 'error'], default='info',
        help="Set minimum log level. (default: info)")
    output_group.add_argument('--prefix', action='store_true',
        help="Add log level to stderr logging output.")

    # Input/output directory
    argparser.add_argument('mnistdir',
        help="Directory to mnist dataset")
    argparser.add_argument('checkpointdir',
        help="Path to temporary checkpoint directory")
    argparser.add_argument('--savedir',
        help="Directory to save the neural network to")

    argparser.add_argument('--epochs', type=int, default=1,
      help="Number of epochs to train for (default 1)")

    args=argparser.parse_args()
 
    # Map argument values to numeric log level
    logging_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARN,
        'error': logging.ERROR,
    }
 
    # Setup logging before anything else.
    fmt_normal="%(message)s"
    # Longest pre-defiend log level is 'CRITICAL', which is 8 characters.
    fmt_prefix="[ %(levelname)-8s ] %(message)s"
 
    # Log to stdout
    log_handler = logging.StreamHandler()
    if args.prefix:
        log_handler.setFormatter(logging.Formatter(fmt=fmt_prefix))
    else:
        log_handler.setFormatter(logging.Formatter(fmt=fmt_normal))
    logger.addHandler(log_handler)
 
    logger.setLevel(logging_levels[args.level])
 
    logger.debug("Logging initialised.")
    logger.debug("Command line arguments: %s", args)

    data = load_mnist(args.mnistdir)
    logger.info("Loaded %d training images, %d training labels, %d test images and %d test labels",
        data['train'].shape[0], data['trnlabl'].size, data['test'].shape[0],
        data['tstlabl'].size)

    network = build_nn(args.checkpointdir)
    train_nn(network, data, args.epochs)
    validate_nn(network, data)
    if args.savedir:
        logger.info("Saving network to: %s", args.savedir)
        save_nn(network, args.savedir)
    else:
        logger.info("Not saving network (see --savedir to enable).")

