import time
from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.slim.python.slim.nets import resnet_utils

from utils import preprocessing

_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4


def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, reuse, depth=256):
    """Atrous Spatial Pyramid Pooling.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
        the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      is_training: A boolean denoting whether the input is for training.
      depth: The depth of the ResNet unit output.

    Returns:
      The atrous spatial pyramid pooling output.
    """
    with tf.variable_scope("aspp", reuse = reuse):
        if output_stride not in [8, 16]:
            raise ValueError('output_stride must be either 8 or 16.')

        atrous_rates = [6, 12, 18]
        if output_stride == 8:
            atrous_rates = [2 * rate for rate in atrous_rates]

        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            with arg_scope([layers.batch_norm], is_training=is_training):
                inputs_size = tf.shape(inputs)[1:3]
                # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
                # the rates are doubled when output stride = 8.
                conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
                conv_3x3_1 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[0],
                                                      scope='conv_3x3_1')
                conv_3x3_2 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[1],
                                                      scope='conv_3x3_2')
                conv_3x3_3 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[2],
                                                      scope='conv_3x3_3')

                # (b) the image-level features
                with tf.variable_scope("image_level_features"):
                    # global average pooling
                    image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
                    # 1x1 convolution with 256 filters( and batch normalization)
                    image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1,
                                                             scope='conv_1x1')
                    # bilinearly upsample features
                    image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

                net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3,
                                name='concat')
                net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

                return net




class Model(object):
    def __init__(self, name, params, mode=tf.estimator.ModeKeys.TRAIN):

        self.name = name
        self.params = params
        self.mode = mode
        # model output
        self.output_dict = OrderedDict()
        self.print_dict = OrderedDict()
        self.loss = None
        self.metrics = None
        self.train_op = None
        self.metrics_init_op = None
        self.update_metrics_op = None
        self.summary_op = None

        self.train_var_list = None


    def inference(self, inputs, reuse=None, data_format='channels_last'):
        if len(self.output_dict) > 0:
            return
        is_training = self.mode == tf.estimator.ModeKeys.TRAIN

        # input feature  ========================

        if data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])


        # backbone model =========================
        base_architecture = self.params['base_architecture']
        if base_architecture not in ['resnet_v2_50', 'resnet_v2_101']:
            raise ValueError("'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_101'.")

        if base_architecture == 'resnet_v2_50':
            base_model = resnet_v2.resnet_v2_50
        else:
            base_model = resnet_v2.resnet_v2_101

        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=_BATCH_NORM_DECAY)):
            logits, end_points = base_model(inputs,
                                            num_classes=None,
                                            is_training=is_training,
                                            global_pool=False,
                                            reuse = reuse,
                                            output_stride=self.params['output_stride'])

        if is_training:
            exclude = [base_architecture + '/logits', '/global_step']
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
            tf.train.init_from_checkpoint(self.params['pre_trained_model'],
                                          {v.name.split(':')[0]: v for v in variables_to_restore})

        # ASPP and upsample ==========================
        resnet_out = end_points[base_architecture + '/block4']
        net = atrous_spatial_pyramid_pooling(resnet_out, self.params['output_stride'],
                                             self.params['batch_norm_decay'], is_training, reuse)

        inputs_size = tf.shape(inputs)[1:3]
        with tf.variable_scope("upsampling_logits", reuse = reuse):
            net = layers_lib.conv2d(net, self.params['num_classes'], [1, 1],
                                    activation_fn=None, normalizer_fn=None,
                                    scope='conv_1x1')
            logits = tf.image.resize_bilinear(net, inputs_size, name='upsample')  # [B, h, w, classes]

        pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)  # [B, h, w, 1]

        # generate the mask in RGB format, shape [B, h, w, 3]
        pred_masks = tf.py_func(preprocessing.decode_labels,
                                         [pred_classes, self.params['batch_size'], self.params['num_classes']],
                                         tf.uint8)

        self.output_dict['logits'] = logits
        self.output_dict['pred_classes'] = pred_classes
        self.output_dict['pred_masks'] = pred_masks

        if not self.params['freeze_batch_norm']:
            self.train_var_list = [v for v in tf.trainable_variables()]
        else:
            self.train_var_list = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]


    def get_loss(self, labels):

        # process ground truth label, only keep valid ones
        labels = tf.squeeze(labels, axis=3)  # reduce the channel dimension.
        labels_flat = tf.reshape(labels, [-1, ])
        valid_indices = tf.to_int32(labels_flat <= self.params['num_classes'] - 1)
        valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

        logits_by_num_classes = tf.reshape(self.output_dict['logits'], [-1, self.params['num_classes']])
        valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]

        preds_flat = tf.reshape(self.output_dict['pred_classes'], [-1, ])
        valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
        confusion_matrix = tf.confusion_matrix(valid_labels, valid_preds, num_classes=self.params['num_classes'])

        self.output_dict['valid_preds'] = valid_preds
        self.output_dict['valid_labels'] = valid_labels
        self.output_dict['confusion_matrix'] = confusion_matrix

        cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=valid_logits, labels=valid_labels)
        # Create a tensor named cross_entropy for logging purposes.
        #tf.identity(cross_entropy, name='cross_entropy')
        tf.summary.scalar('cross_entropy', cross_entropy)

        # Add weight decay to the loss.
        with tf.variable_scope("total_loss"):
            loss = cross_entropy + self.params.get('weight_decay', _WEIGHT_DECAY) * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.train_var_list])
            # loss = tf.losses.get_total_loss()  # obtain the regularization losses as well
        self.loss = loss


    def train(self):
        if self.train_op is not None or self.mode != tf.estimator.ModeKeys.TRAIN:
            return

        if self.params['learning_rate_policy'] == 'piecewise':
            # Scale the learning rate linearly with the batch size. When the batch size
            # is 128, the learning rate should be 0.1.
            initial_learning_rate = 0.1 * self.params['batch_size'] / 128
            batches_per_epoch = self.params['num_train'] / self.params['batch_size']
            # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
            boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
            values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
            learning_rate = tf.train.piecewise_constant(
                tf.train.get_or_create_global_step(), boundaries, values)
        elif self.params['learning_rate_policy'] == 'poly':
            learning_rate = tf.train.polynomial_decay(
                self.params['initial_learning_rate'],
                tf.train.get_or_create_global_step(),
                self.params['max_iter'], self.params['end_learning_rate'], power=self.params['power'])
        else:
            raise ValueError('Learning rate policy must be "piecewise" or "poly"')
        self.print_dict['lr'] = learning_rate

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=self.params['momentum'])

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=tf.train.get_or_create_global_step(),
                                               var_list=self.train_var_list)

    def metrics_n_summary(self):

        # streaming loss and mean_iou
        with tf.variable_scope(('train' if self.mode == tf.estimator.ModeKeys.TRAIN else 'eval') + '_metrics'):
            self.metrics = {
                'loss': tf.metrics.mean(self.loss),
                'px_acc': tf.metrics.accuracy(self.output_dict['valid_labels'], self.output_dict['valid_preds']),
                'mean_iou':tf.metrics.mean_iou(self.output_dict['valid_labels'], self.output_dict['valid_preds'],
                                               self.params['num_classes'])
                       }
            self.update_metrics_op = tf.group(*[op for _, op in self.metrics.values()])
            metrics_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                      scope=('train' if self.mode == tf.estimator.ModeKeys.TRAIN else 'eval')+"metrics")
            self.metrics_init_op = tf.variables_initializer(metrics_variables)

        summaries = []
        # for summary use mean_iou for current batch
        mean_iou = self.compute_mean_iou(self.metrics['mean_iou'][1])
        self.print_dict['loss'] = self.loss      # loss for current batch
        self.print_dict['mean_iou'] = mean_iou   # mean_iou for current batch
        with tf.name_scope('train' if self.mode == tf.estimator.ModeKeys.TRAIN else 'eval'):
            summaries.append(tf.summary.scalar('loss', self.loss))
            summaries.append(tf.summary.scalar('mean_iou', mean_iou))
        self.summary_op = tf.summary.merge(summaries)


    def compute_mean_iou(self, total_cm, name='mean_iou'):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
        sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
        cm_diag = tf.to_float(tf.diag_part(total_cm))
        denominator = sum_over_row + sum_over_col - cm_diag

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(tf.cast(
            tf.not_equal(denominator, 0), dtype=tf.float32))

        # If the value of the denominator is 0, set it to 1 to avoid
        # zero division.
        denominator = tf.where(
            tf.greater(denominator, 0),
            denominator,
            tf.ones_like(denominator))
        iou = tf.div(cm_diag, denominator)

        for i in range(self.params['num_classes']):
            tf.identity(iou[i], name='train_iou_class{}'.format(i))
            tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

        # If the number of valid entries is 0 (no classes) we return 0.
        result = tf.where(
            tf.greater(num_valid_entries, 0),
            tf.reduce_sum(iou, name=name) / num_valid_entries,
            0)
        return result