import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_learning_phase(0)
K.set_session(sess)

import time
import numpy as np

import keras.optimizers
from keras.layers import Embedding, Dense, LSTM, TimeDistributed
from keras.models import Sequential
size = 4
model = Sequential()
input_shape=(1, 3)
model.add(LSTM(size, return_sequences=True, consume_less='gpu', input_shape=input_shape))
model.add(TimeDistributed(Dense(1, activation='sigmoid'), input_shape=(None, size)))

tf.train.write_graph(sess.graph.as_graph_def(), '.', 'lstm.pbtxt', as_text=True)
tf.train.write_graph(sess.graph.as_graph_def(), '.', 'lstm.pb', as_text=False)
print model.input
print model.output
print [ str(o.name) for o in sess.graph.get_operations() if o.type == 'Assign' ]
print model.get_weights()
