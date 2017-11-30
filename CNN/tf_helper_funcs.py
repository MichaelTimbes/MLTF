"""
Written by: Michael Timbes
Acknowledgements: Hvass-Labs ()
"""
import tensorflow as tf
import numpy as np

def form_weights(shape, p_dist = 0.05):
    return tf.Variable(tf.truncated_normal(shape, stddev = p_dist))

def form_biases(length, p_dist = 0.05):
    return tf.Variable(tf.constant(p_dist, shape=[length]))

def shape_FC_to_Conv(input_matrix, input_channels, filter_size, num_filters, use_pooling=True, pooling_filter = [1,2,2,1]):
    return tf.layers.conv2d(inputs = input_matrix, filters = num_filters, kernel_size = [filter_size, filter_size],padding="same", activation=tf.nn.relu)

def new_conv2d_layer(input_m, num_filters, kernel_s, activation_type = tf.nn.relu):
    return tf.layers.conv2d(inputs = input_m,
                           filters= num_filters,
                           kernel_size= [kernel_s, kernel_s],
                           padding= "same",
                           activation= activation_type)

def new_max_pool(input_m,p_size,sride=2):
    return tf.layers.max_pooling2d(inputs=input_m,pool_size=[p_size,p_size],strides=sride)

def new_conv_layer(input_matrix, input_channels, filter_size, num_filters, use_pooling=True, pooling_filter = [1,2,2,1], Standard_RELU_ = True):
    
    filter_shape = [filter_size, filter_size, input_channels, num_filters]
    
    weights = form_weights(shape= filter_shape)
    
    biases = form_biases(length=num_filters)
    
    conv_layer = tf.nn.conv2d(input = input_matrix, filter = weights, strides = [1, 1, 1, 1], padding = 'SAME')
    conv_layer += biases
    
    if use_pooling:
        conv_layer = tf.nn.max_pool(value = conv_layer, ksize = pooling_filter, strides = pooling_filter, padding = 'SAME')
    
    if Standard_RELU_:    
        conv_layer = tf.nn.relu(conv_layer)
    else:
        conv_layer = tf.nn.crelu(conv_layer)
        
    return conv_layer, weights

def flatten_layer(layer, num_features= 0 ):
    
    layer_shape = layer.get_shape()
    # Assuming [-1,prev_layer_size,prev_layer_size,prev_num_filters]
    num_features = layer_shape[1:4].num_elements()
    
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

def fc_layer(input_v, num_inputs, num_outputs, relu_on = True): 

    # Create new weights and biases.
    weights = form_weights(shape=[num_inputs, num_outputs])
    biases = form_biases(length=num_outputs)

    fc_layer = tf.matmul(input_v, weights) + biases

    # Use ReLU?
    if relu_on:
        fc_layer = tf.nn.relu(fc_layer)

    return fc_layer