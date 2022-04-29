import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy import integrate

def smoothmax(x, a):
    return tf.reduce_sum(tf.math.multiply(x,  tf.math.exp(x*a)), axis=-1) / tf.reduce_sum(tf.math.exp(x*a), axis=-1)

def fit_bell_shaped(T, C, N, epochs, loc, scale, optimizer, input_activation, 
                    output_activation, rate_regularization, weight_regularization, tau=0.1, final_state_regularization=0, smoothmax_param=3, recurrent_init_var=None, input_init_var=None):
    '''
    Parameters:
        T: number of time bins
        C: Number of hidden layer units
        N: Number of training examples
        epochs: number of training epochs
        loc: mode of velocity profile
        scale: width (sigma) of velocity profile
    '''
    x = [] # input layers
    h_a = [] # hidden layer activations
    h_r = [] # hidden layer rates
    o = [] # output layers

    def activation(x):
        return tf.math.tanh(x) * (tf.math.sign(x) + 1)/2  # rectified tanh

    dt = 0.01
    n_outputs = 4
    x = tf.keras.Input(shape=(2,))
    weight_regularizer = tf.keras.regularizers.l2(weight_regularization)
    if input_init_var == None:
        input_initializer = "glorot_uniform"
    else:
        input_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=input_init_var**2/C)

    if recurrent_init_var == None:
        recurrent_initializer = "glorot_uniform"
    else:    
        recurrent_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=recurrent_init_var**2/C)

    input_layer = tf.keras.layers.Dense(C, activation=input_activation, name='input_weights', kernel_regularizer=weight_regularizer, kernel_initializer=input_initializer)
    output_layer = tf.keras.layers.Dense(n_outputs, activation=output_activation, name='output_weights', kernel_regularizer=weight_regularizer, kernel_initializer='zeros')
    recurrent_layer = tf.keras.layers.Dense(C, activation='linear', name='recurrent_weights', use_bias=False, kernel_initializer=recurrent_initializer)
    if False:#learn_init:
        #add_initial_state = tf.keras.layers.Lambda(lambda x: x + K.variable(np.random.randn(1,C).astype(np.float32), 
            #                                           name='initial_state'))                                           
        h_0 = tf.Variable(np.random.randn(1,C).astype(np.float32), trainable=True, shape=(None,C), name='initial_state') #learnable initial state
        h_a_from_x = input_layer(x)
        h_a_from_h_r = recurrent_layer(h_0) 
        h_a_from_h_a = (1 - dt/tau) * h_a
        h_a = h_a_from_x + h_a_from_h_r + h_a_from_h_a
        h_a = add_initial_state(input_layer(x)) # hidden unit activation
    else:
        h_a = input_layer(x)

    h_r = activation(h_a)
    o =  output_layer(h_r)
    all_inputs = [x]
    all_outputs = [tf.expand_dims(o, axis=-1)]
    all_rates = [tf.expand_dims(h_r, axis=-1)]
    for i in range(1,T):
        x = tf.keras.Input(shape=(2,))
        h_a_from_x = input_layer(x)
        h_a_from_h_r = recurrent_layer(h_r)
        h_a_from_h_a = (1 - dt/tau) * h_a
        h_a = h_a_from_x + h_a_from_h_r + h_a_from_h_a
        h_r = activation(h_a)
        o = output_layer(h_r)
        all_inputs.append(x)
        all_outputs.append(tf.expand_dims(o, axis=-1))
        all_rates.append(tf.expand_dims(h_r, axis=-1))

    all_outputs = tf.keras.layers.concatenate(all_outputs)
    all_rates = tf.keras.layers.concatenate(all_rates)
    model = tf.keras.Model(all_inputs, [all_outputs, all_rates], name='test_rnn')
    loss_fn = tf.keras.losses.MeanSquaredError()

    theta_train_1 = np.random.rand(N,1) * 2 *np.pi
    theta_train_2 = np.random.rand(N,1) * 2 *np.pi
    #x_train = [np.hstack((np.cos(theta_train), np.sin(theta_train))) for i in range(T)]
    x_train = [np.hstack((np.cos(theta_train_1), np.sin(theta_train_1))) for i in range(T//2)]
    x_train += [np.hstack((np.cos(theta_train_1)+np.cos(theta_train_2), np.sin(theta_train_1) + np.sin(theta_train_2))) for i in range(T//2)]
    velocity_profile = np.sum([norm.pdf(np.arange(T),loc=l,scale=s) for l,s in zip(loc,scale)], axis=0)
    y_vel = np.transpose(np.array(x_train), (1,2,0)) * velocity_profile
    y_pos = np.concatenate([np.zeros((N,2,1)),integrate.cumtrapz(y_vel, dx=0.01, axis=2)],axis=2)
    y_pos *= np.mean(np.abs(y_vel))/np.mean(np.abs(y_pos))
    y_train = np.concatenate([y_vel, y_pos],axis=1)

    rate_train = np.zeros((N,C,T)) #padding zeros for rate outputs
    TBCallback = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
            write_graph=True, write_images=True)

    def rate_regularizer(y_true, y_pred):
        full_rate = tf.reduce_mean(y_pred, axis=(-2,-1))
        final_state = smoothmax(y_pred[:,:,-1], smoothmax_param)
        return rate_regularization * full_rate + final_state_regularization * final_state
        
    model.compile(optimizer=optimizer, loss=[loss_fn, rate_regularizer])
    #tf.keras.utils.plot_model(model, 'rnn_plot.png')

    model.fit(x_train, [y_train, rate_train], epochs=epochs, verbose=0, callbacks=[TBCallback])
    predicted,rates = model(x_train)

    return predicted, rates, y_train, model

rate_regularization = 0
weight_regularization = 1e-5

C=200
epochs = 100
T = 40
N = 300
loc = [.25*T, .75*T]
scale = [.05*T, .05*T]
optimizer = 'adam'
input_activation = 'linear'
output_activation='linear'
input_init_var = 1
recurrent_init_var = 2
input_noise=0.00
predicted, rates, y_train, model = fit_bell_shaped(T, C, N, epochs, loc, scale, optimizer, input_activation, 
                   output_activation, rate_regularization, weight_regularization, final_state_regularization=0, smoothmax_param=3, recurrent_init_var=None, input_init_var=None)

from scipy import integrate