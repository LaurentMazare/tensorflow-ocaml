import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
  tf_c1 = tf.constant([ 1, 2, 3 ], dtype=tf.float32)
  tf_c2 = tf.placeholder("float", shape=[ 3 ], name="x")
  tf_res = tf_c1 + tf_c2

tf.train.write_graph(graph.as_graph_def(), '.', 'load.pbtxt', as_text=True)
tf.train.write_graph(graph.as_graph_def(), '.', 'load.pb', as_text=False)

