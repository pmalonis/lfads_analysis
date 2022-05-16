import tensorflow as tf
import numpy as np
from scipy.stats import norm

T = 40 # total time 
x = [] # input layers
h_a = [] # hidden layer activations
h_r = [] # hidden layer rates
o = [] # output layers

dt = 0.01
tau = 0.1
N = 200
x = tf.keras.Input(shape=(1,))
input_layer = tf.keras.layers.Dense(N, activation='linear')
output_layer = tf.keras.layers.Dense(1, activation='linear')
recurrent_layer = tf.keras.layers.Dense(N, activation='linear')
h_a = input_layer(x)
h_r = tf.math.tanh(h_a)
o =  output_layer(h_r)
all_inputs = [x]
all_outputs = [o]
for i in range(1,T):
    x = tf.keras.Input(shape=(1,), batch_size=1)
    h_a_from_x = input_layer(x)
    h_a_from_h_r = recurrent_layer(h_r)
    h_a_from_h_a = (1 - dt/tau) * h_a # 1 comes from Euler step. Second term is leak (this is folded into weights in other terms) 
    h_a = h_a_from_x + h_a_from_h_r + h_a_from_h_a
    h_r = tf.math.tanh(h_a)
    o = output_layer(h_r)
    all_inputs.append(x)
    all_outputs.append(o)

#all_inputs_tensor = tf.keras.layers.concatenate(all_inputs)
#all_outputs_tensor = tf.keras.layers.concatenate(all_outputs)
all_outputs = tf.keras.layers.concatenate(all_outputs)
model = tf.keras.Model(all_inputs, all_outputs, name='test_rnn')
loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam', loss=loss_fn)
# tf.keras.utils.plot_model(model, 'rnn_plot.png')

k = 300
x_train = [np.ones((k,1)) + np.random.randn(k,1)*0.01 for i in range(T)]
peak = T*0.35
scale = T * 0.08
y_train = np.ones((k,T)) * norm.pdf(np.arange(40),loc=peak,scale=scale)

model.fit(x_train, y_train, epochs=30)