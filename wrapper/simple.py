import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
  tf_c1 = tf.constant([ 1 ], dtype=tf.float32)
  tf_c2 = tf.constant([ 2 ], dtype=tf.float32)
  tf_res = tf_c1 + tf_c2

txt = str(graph.as_graph_def())
with open('simple.pbtxt', 'w') as f: f.write(txt)

