"""Train a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import deeplab_model
from utils import preprocessing
from tensorflow.python import debug as tf_debug
from Flags import FLAGS

import shutil

_NUM_CLASSES = 21
_HEIGHT = 513
_WIDTH = 513
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

_POWER = 0.9
_MOMENTUM = 0.9

_BATCH_NORM_DECAY = 0.9997

_NUM_IMAGES = {
    'train': 10582,
    'validation': 1449,
}

_TRAIN_SIZE = 10580  # assume save 2 checkpoints per epoch

def get_filenames(is_training, data_dir):
  """Return a list of filenames.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: path to the the directory containing the input data.

  Returns:
    A list of file names.
  """
  if is_training:
    return [os.path.join(data_dir, 'voc_train.record')]
  else:
    return [os.path.join(data_dir, 'voc_val.record')]


def parse_record(raw_record):
  """Parse PASCAL image and label from a tf record."""
  keys_to_features = {
      'image/height':
      tf.FixedLenFeature((), tf.int64),
      'image/width':
      tf.FixedLenFeature((), tf.int64),
      'image/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
      tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'label/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'label/format':
      tf.FixedLenFeature((), tf.string, default_value='png'),
  }

  parsed = tf.parse_single_example(raw_record, keys_to_features)

  # height = tf.cast(parsed['image/height'], tf.int32)
  # width = tf.cast(parsed['image/width'], tf.int32)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]), _DEPTH)
  image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
  image.set_shape([None, None, 3])

  label = tf.image.decode_image(
      tf.reshape(parsed['label/encoded'], shape=[]), 1)
  label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
  label.set_shape([None, None, 1])

  return image, label


def preprocess_image(image, label, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Randomly scale the image and label.
    image, label = preprocessing.random_rescale_image_and_label(
        image, label, _MIN_SCALE, _MAX_SCALE)

    # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
    image, label = preprocessing.random_crop_or_pad_image_and_label(
        image, label, _HEIGHT, _WIDTH, _IGNORE_LABEL)

    # Randomly flip the image and label horizontally.
    image, label = preprocessing.random_flip_left_right_image_and_label(
        image, label)

    image.set_shape([_HEIGHT, _WIDTH, 3])
    label.set_shape([_HEIGHT, _WIDTH, 1])

  image = preprocessing.mean_image_subtraction(image)

  return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
  dataset = dataset.flat_map(tf.data.TFRecordDataset)    # flat_map make sure: order of the dataset stays the same

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
      lambda image, label: preprocess_image(image, label, is_training))
  dataset = dataset.prefetch(batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels


def main(args):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  if FLAGS.clean_model_dir:
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_steps=5000)
  model = tf.estimator.Estimator(
      model_fn=deeplab_model.deeplabv3_model_fn,
      model_dir=FLAGS.model_dir,
      config=run_config,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': FLAGS.batch_size,
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': FLAGS.pre_trained_model,
          'batch_norm_decay': _BATCH_NORM_DECAY,
          'num_classes': _NUM_CLASSES,
          'tensorboard_images_max_outputs': FLAGS.tensorboard_images_max_outputs,
          'weight_decay': FLAGS.weight_decay,
          'learning_rate_policy': FLAGS.learning_rate_policy,
          'num_train': _NUM_IMAGES['train'],
          'initial_learning_rate': FLAGS.initial_learning_rate,
          'max_iter': FLAGS.max_iter,
          'end_learning_rate': FLAGS.end_learning_rate,
          'power': _POWER,
          'momentum': _MOMENTUM,
          'freeze_batch_norm': FLAGS.freeze_batch_norm,
          'initial_global_step': FLAGS.initial_global_step
      })

  for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = {
      'learning_rate': 'learning_rate',
      'cross_entropy': 'cross_entropy',
      'train_px_accuracy': 'train_px_accuracy',
      'train_mean_iou': 'train_mean_iou',
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    train_hooks = [logging_hook]
    eval_hooks = None

    if FLAGS.debug:
      debug_hook = tf_debug.LocalCLIDebugHook()
      train_hooks.append(debug_hook)
      eval_hooks = [debug_hook]

    tf.logging.info("Start training=================================================================.")
    model.train(
        input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=train_hooks,
        steps=1  # For debug
    )

    tf.logging.info("Start evaluation=================================================================.")
    # Evaluate the model and print results
    eval_results = model.evaluate(
        # Batch size must be 1 for testing because the images' size differs
        input_fn=lambda: input_fn(False, FLAGS.data_dir, 1),
        hooks=eval_hooks,
        # steps=1  # For debug
    )
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
