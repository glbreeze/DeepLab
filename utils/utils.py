import datetime

import tensorflow as tf
from tensorflow.python.ops.variables import global_variables


def init_uninit_varaibles(session):
    uninit_vars = session.run(tf.report_uninitialized_variables(global_variables()))
    vars_list = []
    for var_name in uninit_vars:
        temp = [var for var in tf.global_variables() if var.op.name == var_name][0]
        vars_list.append(temp)
    session.run(tf.variables_initializer(vars_list))


def get_logger(name):
    import logging, sys
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def print_info(logger, step, infos):
    time_str = datetime.datetime.now().isoformat()
    logger.info("{}: step {}, {}".format(time_str, step, ','.join(['{}={:g}'.format(a, b) for a, b in infos])))

