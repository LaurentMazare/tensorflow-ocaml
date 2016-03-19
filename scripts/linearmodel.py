import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

image_dim = 28 * 28
label_count = 10
graph = tf.Graph()
with graph.as_default():
  x = tf.placeholder("float", shape=[None, image_dim])
  y_ = tf.placeholder("float", shape=[None, label_count])
  W = tf.Variable(tf.zeros([ image_dim, label_count ]))
  b = tf.Variable(tf.zeros([ label_count ]))
  y = tf.nn.softmax(tf.matmul(x, W) + b)
#  cross_entropy = - tf.reduce_mean(y_ * tf.log(y))
  err = tf.nn.l2_loss(y-y_)
  optimizer = tf.train.GradientDescentOptimizer(0.4).minimize(err)
#  optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  tf.train.write_graph(graph.as_graph_def(), '/tmp/tf', 'linearmodel.pbtxt', as_text=True)

  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)
  for i in range(1000):
    print sess.run([ err ], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    sess.run([ optimizer ], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
