import tensorflow as tf
c = 0
file_name = '../dataset/voc_train.record'

sum(1 for _ in tf.python_io.tf_record_iterator(file_name))

# total number of training samples : 10580