import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
  tf_c1 = tf.constant([ 1 ], dtype=tf.float32)
  tf_c2 = tf.Variable(tf.zeros([ 1 ]))
  tf_res = tf_c1 + tf_c2
  ops = tf.train.GradientDescentOptimizer(0.4).minimize(tf_res)

tf.train.write_graph(graph.as_graph_def(), '.', 'simple.pbtxt', as_text=True)
tf.train.write_graph(graph.as_graph_def(), '.', 'test.pb', as_text=False)

