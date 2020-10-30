"""Evaluate a DeepLab v3 model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from PIL import Image

import tensorflow as tf

from utils import preprocessing
from utils import dataset_util

import numpy as np
import timeit
from flags import FLAGS
from model import Model
from utils.utils import *
from hyper_params import *

logger = get_logger(__name__)
_NUM_CLASSES = 21


def evaluate_sess(sess, model, data_init_op,  num_steps, writer=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model: (dict) contains the graph operations or nodes needed for training
        data_init_op: initializer for input_fn()
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
    """
    update_metrics = model.update_metrics_op
    eval_metrics = model.metrics  # include 1. loss, 2. px_acc 3. mean_iou (streaming mean_iou, confusion matrix)
    global_step = tf.train.get_global_step()

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(data_init_op)
    sess.run(model.metrics_init_op)

    # compute metrics over the dataset
    for _ in range(num_steps):
        sess.run(update_metrics)

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)

    # Add summaries manually to writer at global_step_val
    if writer is not None:
        global_step_val = sess.run(global_step)
        smy_items = []
        for tag, val in metrics_val.items():
            smy_items.append( tf.Summary.Value(tag='eval/'+tag, simple_value=val) )
        summ = tf.Summary(value=smy_items)
        writer.add_summary(summ, global_step_val)

    return metrics_val




def main(args):
    # fetch data
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    examples = dataset_util.read_examples_list(FLAGS.evaluation_data_list)  # list of file names
    image_files = [os.path.join(FLAGS.image_dir, filename) + '.jpg' for filename in examples]
    label_files = [os.path.join(FLAGS.eval_label_dir, filename) + '.png' for filename in examples]
    features, labels = preprocessing.eval_input_fn(image_files, label_files)

    # set up the model
    eval_model = Model(name='eval', params=params, mode=tf.estimator.ModeKeys.EVAL)
    eval_model.inference(features, reuse=False)
    eval_model.get_loss(labels)
    eval_model.metrics_n_summary()

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options,
                            log_device_placement=False,
                            allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        # Manually load the latest checkpoint
        saver.restore(sess, FLAGS.test_model_dir)
        sess.run(eval_model.metrics_init_op)  # need to init the local variable for calculate the metrics

        step = 0
        out_file = open('../dataset/VOCdevkit/pred/eval_metric.txt', 'w')
        while True:
            try:
                output, _, metrics, current_metrics = sess.run([eval_model.output_dict, eval_model.update_metrics_op,
                                                                eval_model.metrics, eval_model.print_dict])
                step += 1
                # output the predicted mask
                pred_mask = np.squeeze(output['pred_masks'], axis=0)
                mask_im = Image.fromarray(pred_mask)
                mask_im.save(os.path.join('../dataset/VOCdevkit/pred/pred_mask/', 'mask'+str(step)+'.jpg'))

                # evaluation metrics
                metrics_values = {k: v[0] for k, v in metrics.items()}
                logger.info("step {} current metrics values {}".format(step, metrics_values))

                # metrics for current batch
                current_mean_iou, current_loss = current_metrics['mean_iou'], current_metrics['loss']
                pixel_acc = len(np.where(output['valid_preds'] == output['valid_labels'])[0]
                                )/len(output['valid_labels'])
                out_file.write('step {}, mean_iou {}, loss {}, pixel_acc {} \n'.format(
                    step, round(current_mean_iou, 4), round(current_loss, 2), round(pixel_acc, 4)
                ))

            except tf.errors.OutOfRangeError:
                print('done with evaluation')
                break

        out_file.close()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
