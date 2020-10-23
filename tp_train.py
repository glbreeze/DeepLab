import os
from sys import exit

from Flags import FLAGS
from HyperParams import params
from InputFn import input_train_fn, input_eval_fn
from Model import Model, get_layout_summary
from Util import *

logger = get_logger(__name__)


def check_need_parameter():
    pass


def invoke_test_shell(out_dir, checkpoint_prefix, step):
    gpu_id = FLAGS.do_test_gpu_id
    model_path = '{}-{}'.format(checkpoint_prefix, step)
    command = 'nohup sh run_test.sh {} {} {} {} &'.format(gpu_id, model_path, out_dir, step)
    logger.info("invoke external command\n" + command)
    os.system(command)


def invoke_test(test_out_dir, checkpoint_prefix, step):
    if not os.path.exists(test_out_dir):
        os.mkdir(test_out_dir)
    gpu_id = FLAGS.do_test_gpu_id
    test_data_dir = FLAGS.test_data_dir
    model_path = '{}-{}'.format(checkpoint_prefix, step)
    out_data_name = 'test_output_{}'.format(step)
    out_data_path = os.path.join(test_out_dir, out_data_name)
    out_data_log = '{}.log'.format(out_data_name)
    out_data_log_path = os.path.join(test_out_dir, out_data_log)
    command = 'CUDA_VISIBLE_DEVICES={} nohup python TestModel.py --test_data_dir {} --model_path {} --out_data_path {} > {} 2>&1 &' \
        .format(gpu_id, test_data_dir, model_path, out_data_path, out_data_log_path)
    logger.info("Invoke external command\n" + command)
    os.system(command)


def main(args):
    logger.info(params)

    run_name = FLAGS.run_name
    if run_name is None:
        exit("run_name cannot not be None")
    if os.path.exists(run_name):
        if not FLAGS.do_resume:
            exit("run_name has already exists")

    if FLAGS.learning_rate_type == 'fixed':
        lr = tf.constant(FLAGS.learning_rate, tf.float32, name='lr')
    elif FLAGS.learning_rate_type == 'exponential':
        decay_steps = int(FLAGS.num_per_epoch * FLAGS.num_epochs_per_decay / FLAGS.batch_size)
        lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                        tf.train.get_or_create_global_step(),
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True, name='lr')
    elif FLAGS.learning_rate_type == 'polynomial':
        decay_steps = int(FLAGS.num_per_epoch * FLAGS.num_epochs_per_decay / FLAGS.batch_size)
        lr = tf.train.polynomial_decay(FLAGS.learning_rate,
                                       tf.train.get_or_create_global_step(),
                                       decay_steps,
                                       FLAGS.end_learning_rate,
                                       power=1.0,
                                       cycle=True, name='lr')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                         FLAGS.learning_rate_decay_type)

    train_features, train_labels = input_train_fn()
    eval_features, eval_labels = input_eval_fn()

    with tf.name_scope(name='Train'):
        train_price_model = Model(name='train', params=params, mode=tf.estimator.ModeKeys.TRAIN)
        train_price_model.inference(train_features)
        if FLAGS.restore_old_model:
            tf.train.init_from_checkpoint(FLAGS.old_model_path,
                                          {train_price_model.name + '/': train_price_model.name + '/'})
        train_price_model.metrics(train_labels)
        train_price_model.train(lr)

    with tf.name_scope(name='Eval'):
        eval_price_model = Model(name='price', params=params, mode=tf.estimator.ModeKeys.EVAL)
        eval_price_model.inference(eval_features, reuse=True)
        eval_price_model.metrics(eval_labels)

    global_step = tf.train.get_or_create_global_step()

    # Saver var_list is None, means saving all variables (including global and local)
    # When restore, will restore train dataset iterator
    saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints,
                           name='saver')

    out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.run_name))
    logger.info("Will writing to {}".format(out_dir))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    eval_summary_dir = os.path.join(out_dir, "summaries", "eval")
    test_out_dir = os.path.abspath(os.path.join(out_dir, "test"))
    if not os.path.exists(test_out_dir):
        os.makedirs(test_out_dir)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options,
                            log_device_placement=False,
                            allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        if FLAGS.do_resume:
            logger.info("Model restore from file: " + FLAGS.resume_model_path)
            saver.restore(sess, FLAGS.resume_model_path)
        elif FLAGS.restore_old_model:
            init_uninit_varaibles(sess)
            sess.run(tf.local_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, sess.graph)
        train_summary_writer.add_summary(get_layout_summary())

        start_time = datetime.datetime.now()
        init_step = tf.train.global_step(sess, global_step)

        do_test_checkpoint_i = 0
        while True:
            try:
                train_result_tuple = sess.run(
                    [train_price_model.train_op, global_step,
                     train_price_model.summary_op, train_price_model.print_dict.values()
                     ]
                )

                summaries = train_result_tuple[2]
                step = tf.train.global_step(sess, global_step)
                info = zip(train_price_model.print_dict.keys(),
                           train_result_tuple[3])
                print_info(logger, step, info)
                train_summary_writer.add_summary(summaries, step)

                if step % 5 == 0:
                    end_time = datetime.datetime.now()
                    seconds = (end_time - start_time).seconds
                    speed = (step - init_step) * 1.0 / seconds
                    instance_speed = (step - init_step) * FLAGS.batch_size * 1.0 / seconds
                    logger.info("Speed: {} steps per second, {} instances per second".format(speed, instance_speed))

                if step % FLAGS.evaluate_every == 0:
                    eval_result_tuple = sess.run(
                        [global_step, eval_price_model.summary_op, eval_price_model.print_dict.values()]
                    )
                    step, summaries = eval_result_tuple[0], eval_result_tuple[1]
                    info = zip(eval_price_model.print_dict.keys(),
                               eval_result_tuple[2])
                    logger.info("Evaluation: ")
                    print_info(logger, step, info)
                    eval_summary_writer.add_summary(summaries, step)

                if step % FLAGS.checkpoint_every == 0 and step >= FLAGS.start_checkpoint:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    logger.info("Saved model checkpoint to {}\n".format(path))
                    if FLAGS.do_test_checkpoint and step > FLAGS.start_test_point:
                        logger.info("Can do test checkpoint with index {}".format(do_test_checkpoint_i))
                        if do_test_checkpoint_i % FLAGS.do_test_checkpoint_every == 0:
                            # invoke_test(test_out_dir, checkpoint_prefix, step)
                            logger.info("Will do test checkpoint with index {}".format(do_test_checkpoint_i))
                            invoke_test_shell(out_dir, checkpoint_prefix, step)
                        do_test_checkpoint_i += 1
                        logger.info("Now test checkpoint index {}".format(do_test_checkpoint_i))
            except Exception as e:
                logger.error(e)
                break

        coord.request_stop()
        coord.join(threads)

        saver.save(sess, checkpoint_prefix + "_latest")


if __name__ == '__main__':
    tf.app.run()