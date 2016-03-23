import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
  tf_c1 = tf.placeholder("float", shape=[ None, 3 ])
  b = tf.Variable(tf.zeros([ 3 ]))
  tf_res = tf.reduce_sum(tf.square(b-tf_c1))
  ops = tf.train.GradientDescentOptimizer(0.4).minimize(tf_res)

tf.train.write_graph(graph.as_graph_def(), '.', 'simple.pbtxt', as_text=True)
tf.train.write_graph(graph.as_graph_def(), '.', 'test.pb', as_text=False)

