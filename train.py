"""Train a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pdb

import os
from utils import preprocessing
#import tensorflow as tf
#from tensorflow.python import debug as tf_debug
#from flags import FLAGS
from hyper_params import *
from model import Model
from evaluate import *
from utils.utils import *
from datetime import datetime
import shutil

logger = get_logger(__name__)


_HEIGHT = 513
_WIDTH = 513
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255
EVALUATE_EVERY = round( NUM_IMAGES['train']*FLAGS.epochs_per_eval/FLAGS.batch_size )
CHECKPOINT_EVERY = round( NUM_IMAGES['train']*FLAGS.epochs_per_ckpt/FLAGS.batch_size )


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

    image = tf.image.decode_image(tf.reshape(parsed['image/encoded'], shape=[]), _DEPTH)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])

    label = tf.image.decode_image(tf.reshape(parsed['label/encoded'], shape=[]), 1)
    label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
    label.set_shape([None, None, 1])

    return image, label


def preprocess_image(image, label, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Randomly scale the image and label.
        image, label = preprocessing.random_rescale_image_and_label(image, label, _MIN_SCALE, _MAX_SCALE)

        # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
        image, label = preprocessing.random_crop_or_pad_image_and_label(image, label, _HEIGHT, _WIDTH, _IGNORE_LABEL)

        # Randomly flip the image and label horizontally.
        image, label = preprocessing.random_flip_left_right_image_and_label(image, label)

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
        # choose shuffle buffer sizes, larger sizes result in better randomness, smaller sizes have better performance.
        # Pascal is a relatively small dataset, we choose to shuffle the full epoch.
        dataset = dataset.shuffle(buffer_size=NUM_IMAGES['train'])

    dataset = dataset.map(parse_record)
    dataset = dataset.map(lambda image, label: preprocess_image(image, label, is_training))
    dataset = dataset.prefetch(batch_size)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator() # dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    return images, labels, iterator_init_op


def main(args):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    if FLAGS.clean_model_dir:
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

    # Batch size must be 1 for testing because the images' size differs
    train_features, train_labels, train_iter_init_op = input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.train_epochs)
    eval_features, eval_labels, eval_iter_init_op = input_fn(False, FLAGS.data_dir, 1, num_epochs=1)

    with tf.name_scope(name='Train'):
        train_model = Model(name='train_model', params=params, mode=tf.estimator.ModeKeys.TRAIN)
        train_model.inference(train_features)
        train_model.get_loss(train_labels)
        train_model.train()
        train_model.metrics_n_summary()

    with tf.name_scope(name='Eval'):
        eval_model = Model(name='eval', params=params, mode=tf.estimator.ModeKeys.EVAL)
        eval_model.inference(eval_features, reuse=True)
        eval_model.get_loss(eval_labels)
        eval_model.metrics_n_summary()

    global_step = tf.train.get_or_create_global_step()

    # Saver var_list is None, means saving all variables (including global and local)
    # When restore, will restore train dataset iterator
    saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints,name='saver')

    logger.info("Will writing to {}".format(FLAGS.model_dir))
    checkpoint_dir = FLAGS.model_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    train_summary_dir = os.path.join(FLAGS.model_dir, "summaries", "train")
    eval_summary_dir = os.path.join(FLAGS.model_dir, "summaries", "eval")

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options,
                            log_device_placement=False,
                            allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        if FLAGS.resume:
            logger.info('model restored form file :{}'+ FLAGS.resume_dir)
            saver.restore(sess, FLAGS.resume_dir)
            sess.run(tf.local_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        # initialize training data iteration
        sess.run(train_iter_init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, sess.graph)
        #train_summary_writer.add_summary(get_layout_summary())

        do_test_checkpoint_i = 0
        while True:
            try:
                train_result_tuple = sess.run([
                    train_model.train_op, global_step,
                    train_model.summary_op, train_model.print_dict
                ])

                summaries = train_result_tuple[2]
                step = tf.train.global_step(sess, global_step)
                infos = zip(train_model.print_dict.keys(),train_result_tuple[3].values())
                logger.info( "step {}, {}".format(step, ','.join(['{}={}'.format(a, b) for a, b in infos])))
                train_summary_writer.add_summary(summaries, step)

                if step%10 == 0:
                    logger.info('current step {} ===== time {} ==='.format(step, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

                if step % EVALUATE_EVERY == 0:
                    num_steps = NUM_IMAGES['validation']
                    logger.info('='*10 + 'start evaluation at {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    eval_metrics = evaluate_sess(sess, eval_model, eval_iter_init_op,  num_steps, writer=eval_summary_writer)
                    metrics_string = " ; ".join("{}: {:06.3f}".format(k, v) for k, v in eval_metrics.items())
                    logger.info('Global_step {}'.format(step) +
                                '='*10 +
                                'Evaluation metrics: ' + '='*20 + '\n' + metrics_string)

                if step % CHECKPOINT_EVERY == 0 and step > NUM_IMAGES['train']*FLAGS.epochs_start_ckpt:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    logger.info("Saved model checkpoint to {}\n".format(path))

            except Exception as e:
                logger.error(e)
                break

        coord.request_stop()
        coord.join(threads)

        saver.save(sess, checkpoint_prefix + "_latest")


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
