import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
  tf_ys = tf.placeholder("float", shape=[ None, 1 ])
  b = tf.Variable(tf.zeros([ 1 ]))
  tf_res = tf.reduce_sum(tf_ys - b)
  ops = tf.train.GradientDescentOptimizer(0.4).minimize(tf_res)

tf.train.write_graph(graph.as_graph_def(), '.', 'simple.pbtxt', as_text=True)
