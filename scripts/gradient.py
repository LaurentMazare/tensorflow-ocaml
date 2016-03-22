import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.ops import functional_ops
graph = tf.Graph()
with graph.as_default():
  tt = tf.constant([4.2])
  def XSquarePlusOne(x):
    ph = tf.placeholder("float", shape=[1])
    return x * x + 1.0

  def XSquarePlusOneGrad(x, dy):
    dx = functional_ops._symbolic_gradient(input=[x, dy],
                                         Tout=[tf.float32],
                                         f="XSquarePlusOne",
                                         name="dx")
    return dx

  f = function.define_function(XSquarePlusOne, {"x": tf.float32})
  g = function.define_function(XSquarePlusOneGrad, {"x": tf.float32,
                                                      "dy": tf.float32})
  epsilon = tf.constant([1.0])
  two = tf.constant([2.0])
  call_f = function.call_function(f, two)
  call_g = function.call_function(g, two, epsilon)

  tf.train.write_graph(graph.as_graph_def(), '/tmp/tfb', 'simple.pbtxt', as_text=True)

  with tf.Session() as sess:
    print sess.run(call_f)
    print sess.run(call_g)
