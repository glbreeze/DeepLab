# -*- coding: utf-8
import time
from collections import OrderedDict

import tensorflow as tf
from tensorboard import summary as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2
from Flags import FLAGS
from tensorflow.estimator import ModeKeys

class Model(object):
    def __init__(self, name, params, mode=tf.estimator.ModeKeys.TRAIN):
        self.name = name
        # hyper
        self.params = params

        self.x_title_length = params['x_title_length']
        self.x_desc_length = params['x_desc_length']
        self.cate_size = params['cate_size']
        self.cate_embed_size = params['cate_embed_size']
        self.op_label_size = params['op_label_size']
        self.op_label_embed_size = params['op_label_embed_size']
        self.tb_cate_size = params['tb_cate_size']
        self.tb_cate_embed_size = params['tb_cate_embed_size']

        self.normal_embed_size = params['normal_embed_size']

        self.y_mean = params['y_mean']
        self.y_std = params['y_std']

        # 未成交商品降价折扣率
        self.gamma = params['gamma']
        assert self.gamma < 1.0
        # 低于gamma*p的惩罚系数
        self.sell_low_factor = params['sell_low_factor']
        # 高于p的惩罚系数
        self.sell_high_factor = params['sell_high_factor']
        # 成交商品的加价折扣率
        self.beta = params['beta']
        assert self.beta > 1.0
        # 低于p的惩罚系数
        self.pay_low_factor = params['pay_low_factor']
        # 高于beta*p的惩罚系数
        self.pay_high_factor = params['pay_high_factor']
        # 没有成交的权重
        self.alpha = params['alpha']
        assert self.alpha < 1.0

        self.dropout_rate = params['dropout_rate']
        self.weight_decay = params['weight_decay']

        self.mode = mode

        # model output
        self.output_dict = OrderedDict()
        self.loss = None
        self.train_op = None
        self.summary_op = None
        self.print_dict = OrderedDict()

    def inference(self, features, reuse=None):
        if len(self.output_dict) > 0:
            return

        with tf.variable_scope(name_or_scope=self.name, values=features.values(), reuse=reuse):
            with tf.variable_scope("text_embed"):
                W = tf.get_variable(name="W", shape=[314685+1, 256],
                                    initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
                title_word_embed = tf.nn.embedding_lookup(W, features['x_title_seg'], name='title_embed')
                desc_word_embed = tf.nn.embedding_lookup(W, features['x_desc_seg'], name='desc_embed')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.weight_decay * tf.nn.l2_loss(W))

            with tf.variable_scope("title_rnn"):
                keep_prob = 1.0 - self.dropout_rate if self.mode == tf.estimator.ModeKeys.TRAIN else 1.0
                cell_f = tf.nn.rnn_cell.BasicLSTMCell(128)
                cell_f = tf.nn.rnn_cell.DropoutWrapper(cell_f, output_keep_prob=keep_prob)
                cell_b = tf.nn.rnn_cell.BasicLSTMCell(128)
                cell_b = tf.nn.rnn_cell.DropoutWrapper(cell_b, output_keep_prob=keep_prob)
                _, (text_embed_f, text_embed_b) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_f, cell_bw=cell_b,
                                                  inputs=title_word_embed,
                                                  sequence_length=features['x_title_seg_len'],
                                                  dtype=tf.float32)
                text_embed = tf.concat((text_embed_f.h, text_embed_b.h), axis=1)

            with tf.variable_scope("desc_rnn"):
                keep_prob = 1.0 - self.dropout_rate if self.mode == tf.estimator.ModeKeys.TRAIN else 1.0
                cell_f = tf.nn.rnn_cell.BasicLSTMCell(128)
                cell_f = tf.nn.rnn_cell.DropoutWrapper(cell_f, output_keep_prob=keep_prob)
                cell_b = tf.nn.rnn_cell.BasicLSTMCell(128)
                cell_b = tf.nn.rnn_cell.DropoutWrapper(cell_b, output_keep_prob=keep_prob)
                _, (desc_embed_f, desc_embed_b) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_f, cell_bw=cell_b,
                                                  inputs=desc_word_embed,
                                                  sequence_length=features['x_desc_seg_len'],
                                                  dtype=tf.float32)
                desc_embed = tf.concat((desc_embed_f.h, desc_embed_b.h), axis=1)

            text_dense = _dense_layer(text_embed, 256, training=self.mode==tf.estimator.ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='text_dense')
            desc_dense = _dense_layer(desc_embed, 256, training=self.mode==tf.estimator.ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='desc_dense')

            img_dense = _dense_layer(features['x_img_feature'], 256, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='img_dense')

            cate_embed = _common_embed(features['x_cate_id'], self.cate_size, self.cate_embed_size, weight_decay=self.weight_decay, scope='cate_embed')
            cate_dense = _dense_layer(cate_embed, 64, training=self.mode==tf.estimator.ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='cate_dense')
            tb_cate_embed = _common_embed(features['x_tb_cate_id'], self.tb_cate_size, self.tb_cate_embed_size, weight_decay=self.weight_decay, scope='tb_cate_embed')
            tb_cate_dense = _dense_layer(tb_cate_embed, 64, training=self.mode==tf.estimator.ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='tb_cate_dense')
            op_label_embed = _common_embed(features['x_op_label'], self.op_label_size, self.op_label_embed_size, weight_decay=self.weight_decay, scope='op_label_embed')
            op_label_dense = _dense_layer(op_label_embed, 64, training=self.mode==tf.estimator.ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='op_label_dense')

            # 'x_bargain', 'x_free_ship', 'x_is_resell',
            embed_x_bargain = _common_embed(features['x_bargain'], 3, 4, weight_decay=self.weight_decay, scope='x_bargain')
            x_bargain_dense = _dense_layer(embed_x_bargain, 4, training=self.mode==tf.estimator.ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='x_bargain_dense')
            embed_x_free_ship = _common_embed(features['x_free_ship'], 3, 4, weight_decay=self.weight_decay, scope='x_free_ship')
            x_free_ship_dense = _dense_layer(embed_x_free_ship, 4, training=self.mode==tf.estimator.ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='x_free_ship_dense')
            x_is_resell = tf.reshape(tf.cast(features['x_is_resell'], tf.float32), (-1, 1))

            # 'u_pub_nums', 'u_pay_nums', 'u_pay_rate', 'uc_pub_nums', 'uc_pay_nums', 'uc_pay_rate'
            embed_u_pub_nums = _common_embed(features['u_pub_nums'], 10, self.normal_embed_size,
                                             weight_decay=self.weight_decay, scope="u_pub_nums_embed")
            u_pub_nums_dense = _dense_layer(embed_u_pub_nums, self.normal_embed_size, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='u_pub_nums_dense')
            embed_u_pay_nums = _common_embed(features['u_pay_nums'], 10, self.normal_embed_size,
                                             weight_decay=self.weight_decay, scope="u_pay_nums_embed")
            u_pay_nums_dense = _dense_layer(embed_u_pay_nums, self.normal_embed_size, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='u_pay_nums_dense')
            embed_u_pay_rate = _common_embed(features['u_pay_rate'], 6, self.normal_embed_size,
                                             weight_decay=self.weight_decay, scope="u_pay_rate_embed")
            u_pay_rate_dense = _dense_layer(embed_u_pay_rate, self.normal_embed_size, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='u_pay_rate_dense')
            embed_uc_pub_nums = _common_embed(features['uc_pub_nums'], 10, self.normal_embed_size,
                                              weight_decay=self.weight_decay, scope="uc_pub_nums_embed")
            uc_pub_nums_dense = _dense_layer(embed_uc_pub_nums, self.normal_embed_size, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='uc_pub_nums_dense')
            embed_uc_pay_nums = _common_embed(features['uc_pay_nums'], 10, self.normal_embed_size,
                                              weight_decay=self.weight_decay, scope="uc_pay_nums_embed")
            uc_pay_nums_dense = _dense_layer(embed_uc_pay_nums, self.normal_embed_size, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='uc_pay_nums_dense')
            embed_uc_pay_rate = _common_embed(features['uc_pay_rate'], 6, self.normal_embed_size,
                                              weight_decay=self.weight_decay, scope="uc_pay_rate_embed")
            uc_pay_rate_dense = _dense_layer(embed_uc_pay_rate, self.normal_embed_size, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='uc_pay_rate_dense')
            embed_utbc_pub_nums = _common_embed(features['utbc_pub_nums'], 10, self.normal_embed_size,
                                                weight_decay=self.weight_decay, scope="utbc_pub_nums_embed")
            utbc_pub_nums_dense = _dense_layer(embed_utbc_pub_nums, self.normal_embed_size, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='utbc_pub_nums_dense')
            embed_utbc_pay_nums = _common_embed(features['utbc_pay_nums'], 10, self.normal_embed_size,
                                                weight_decay=self.weight_decay, scope="utbc_pay_nums_embed")
            utbc_pay_nums_dense = _dense_layer(embed_utbc_pay_nums, self.normal_embed_size, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='utbc_pay_nums_dense')
            embed_utbc_pay_rate = _common_embed(features['utbc_pay_rate'], 6, self.normal_embed_size,
                                                weight_decay=self.weight_decay, scope="utbc_pay_rate_embed")
            utbc_pay_rate_dense = _dense_layer(embed_utbc_pay_rate, self.normal_embed_size, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='utbc_pay_rate_dense')
            all_embed = tf.concat([cate_dense, tb_cate_dense, op_label_dense,
                                   x_bargain_dense, x_free_ship_dense,
                         u_pub_nums_dense, u_pay_nums_dense, u_pay_rate_dense,
                         uc_pub_nums_dense, uc_pay_nums_dense, uc_pay_rate_dense,
                         utbc_pub_nums_dense, utbc_pay_nums_dense, utbc_pay_rate_dense], axis=1, name='all_embed')
            all_embed_dense = _dense_layer(all_embed, out_dim=512, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='all_embed_dense')

            # continuous
            global_price = features['global_price']
            c_price = features['c_price']
            tbc_price = features['tbc_price']
            u_price = features['u_price']
            uc_price = features['uc_price']
            utbc_price = features['utbc_price']
            all_cont = tf.concat([global_price, c_price, tbc_price, u_price, uc_price, utbc_price, x_is_resell], axis=1, name='all_cont')
            all_cont_dense = _dense_layer(all_cont, out_dim=128, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='all_cont_dense')

            concat1 = tf.concat([text_dense, desc_dense, img_dense, all_embed_dense, all_cont_dense], axis=1, name='concat1')

            dense1, dense1_a, dense1_bn, dense1_drop = _dense_layer(concat1, 512, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='dense1', out_all=True)
            self.output_dict['dense1'] = dense1
            self.output_dict['dense1_a'] = dense1_a
            self.output_dict['dense1_bn'] = dense1_bn
            self.output_dict['dense1_drop'] = dense1_drop
            dense2, dense2_a, dense2_bn, dense2_drop = _dense_layer(dense1_drop, 256, training=self.mode==ModeKeys.TRAIN, dropout_ratio=self.dropout_rate, weight_decay=self.weight_decay, scope='dense2', out_all=True)
            self.output_dict['dense2'] = dense2
            self.output_dict['dense2_a'] = dense2_a
            self.output_dict['dense2_bn'] = dense2_bn
            self.output_dict['dense2_drop'] = dense2_drop
            logits = tf.squeeze(tf.layers.dense(dense2_drop, 1, use_bias=True,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                                                bias_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay)),
                                axis=1,
                                name='logits')
            pred_prices = tf.expm1(logits * self.y_std + self.y_mean, name="prices")

        self.output_dict['logits'] = logits
        self.output_dict['pred_prices'] = pred_prices

    def metrics(self, labels):
        if self.loss is not None:
            return
        logits = self.output_dict['logits']
        pay_price = labels['pay_price']
        reg_loss = tf.add_n(tf.losses.get_regularization_losses(scope=tf.contrib.framework.get_name_scope()), name='reg_loss')
        self.print_dict['reg_loss'] = reg_loss
        loss = tf.reduce_mean(_huber_loss((tf.log1p(pay_price) - self.y_mean) / self.y_std, logits), name='loss')
        self.loss = loss + reg_loss
        self.add_summary(labels)



    def train(self, lr):
        if self.train_op is not None or self.mode != tf.estimator.ModeKeys.TRAIN:
            return
        optimizer = tf.train.AdamOptimizer(lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_or_create_global_step())
        if FLAGS.use_ema:
            model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ema_decay)
            minimize_op = self.train_op
            with tf.control_dependencies([minimize_op]):
                self.train_op = ema.apply(model_vars)
        self.print_dict['lr'] = lr

    def add_summary(self, labels):
        summaries = []
        is_pay = labels['is_pay']
        pay_price = labels['pay_price']
        p = tf.boolean_mask(pay_price, is_pay)
        p_hat = tf.boolean_mask(self.output_dict['pred_prices'], is_pay)

        all_rmsle = _rmsle(p, p_hat, name='rmsle')
        with tf.name_scope('train' if self.mode == tf.estimator.ModeKeys.TRAIN else 'eval'):
            summaries.append(summary_lib.scalar('loss', self.loss))
            summaries.append(summary_lib.scalar('rmsle', all_rmsle))
            # ranges = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000),
            #           (5000, 999999)]
            # for min_v, max_v in ranges:
            #     name = '{}_{}_rmsle'.format(min_v, max_v)
            #     tmp_rmsle = _rmsle(p, p_hat, min_v=min_v, max_v=max_v, name=name)
            #     summaries.append(summary_lib.scalar(name, tmp_rmsle))

        self.print_dict['loss'] = self.loss
        self.print_dict['rmsle'] = all_rmsle
        self.summary_op = tf.summary.merge(summaries)


def _rmsle(p, p_hat, min_v=None, max_v=None, name=None):
    with tf.name_scope(name=name):
        if min_v is None:
            return tf.cond(tf.equal(tf.size(p), tf.constant(0, dtype=tf.int32)),
                           lambda: tf.constant(0, dtype=tf.float32),
                           lambda: tf.sqrt(tf.reduce_mean((tf.log1p(p) - tf.log1p(p_hat)) ** 2)))
        else:
            ids = tf.squeeze(tf.where(tf.logical_and(p >= min_v, p < max_v)))
            return tf.cond(tf.equal(tf.size(ids), tf.constant(0, dtype=tf.int32)),
                           lambda: tf.constant(0, dtype=tf.float32),
                           lambda: tf.sqrt(
                               tf.reduce_mean((tf.log1p(tf.gather(p, ids)) - tf.log1p(tf.gather(p_hat, ids))) ** 2)))


def get_layout_summary():
    return summary_lib.custom_scalar_pb(
        layout_pb2.Layout(
            category=[
                layout_pb2.Category(
                    title="metrics",
                    chart=[
                        layout_pb2.Chart(
                            title="losses",
                            multiline=layout_pb2.MultilineChartContent(
                                tag=['train/loss', 'eval/loss']
                            )
                        ),
                        layout_pb2.Chart(
                            title="rmsles",
                            multiline=layout_pb2.MultilineChartContent(
                                tag=['train/rmsle', 'eval/rmsle']
                            )
                        ),
                    ]
                )
            ]
        )
    )


def _common_embed(x, input_size, embed_size, weight_decay=None, scope=None):
    with tf.variable_scope(scope or "embed"):
        W = tf.get_variable(name="W", shape=[input_size, embed_size],
                            initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        if weight_decay is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay * tf.nn.l2_loss(W))
        return tf.nn.embedding_lookup(W, x, name=scope)


def _huber_loss(labels, logits, delta=0.6):
    abs_error = tf.abs(labels - logits)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    return 0.5 * quadratic ** 2 + delta * linear


def _dense_layer(input, out_dim, training=True, dropout_ratio=0.5, weight_decay=None, scope=None, out_all=None):
    with tf.variable_scope(scope if scope is not None else '_Dense'):
        if weight_decay is None:
            dense = tf.layers.dense(input, units=out_dim, use_bias=True)
            dense_a = tf.nn.relu(dense)
        else:
            dense = tf.layers.dense(input, units=out_dim, use_bias=True,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            dense_a = tf.nn.relu(dense)
        dense_bn = tf.layers.batch_normalization(dense_a, training=training)
        dense_drop = tf.layers.dropout(dense_bn, rate=dropout_ratio, training=training)
        if out_all is None or out_all != True:
            return dense_drop
        else:
            return dense, dense_a, dense_bn, dense_drop

