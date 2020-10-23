
import torch
import torch.nn as nn

pixel_shuffle = nn.PixelShuffle(3)
input = torch.randn(1, 9, 4, 4)
output = pixel_shuffle(input)
print(output.size())



import tensorflow as tf

x = tf.constant([[[[1, 2, 3, 15], [4, 5, 6, 16]],
      [[7, 8, 9, 17], [10, 11, 12, 13]]]])
print(x.shape)
y = tf.depth_to_space(x,2)
with tf.Session() as sess:
     z = sess.run(y)
print(z)


import tensorflow as tf

with tf.Session() as sess:
    writer = tf.summary.FileWriter('sum',sess.graph)
    value = 37.0
    summary = tf.Summary(value=[
        tf.Summary.Value(tag="summary_tag", simple_value=value),
        tf.Summary.Value(tag="summary_tag1", simple_value=value),
    ])
    writer.add_summary(summary)