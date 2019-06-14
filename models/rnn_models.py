
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import collections
import hashlib
import numbers
import itertools
import functools
import sets
import inspect
import pickle

from sklearn.model_selection import *
from sklearn.metrics import *

import tensorflow as tf
from tensorflow.contrib.rnn import PhasedLSTMCell, MultiRNNCell, BasicRNNCell
from tensorflow.python.ops import rnn_cell, rnn
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op, dtypes, ops, tensor_shape, tensor_util   
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import * 
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpointable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings("ignore")


# Set of helper functions

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


def padd_data(X, padd_length):
    
    X_padded      = []
    
    for k in range(len(X)):
        
        if X[k].shape[0] < padd_length:
            
            if len(X[k].shape) > 1:
                X_padded.append(np.array(np.vstack((np.array(X[k]), 
                                                    np.zeros((padd_length-X[k].shape[0],X[k].shape[1]))))))
            else:
                X_padded.append(np.array(np.vstack((np.array(X[k]).reshape((-1,1)),
                                                    np.zeros((padd_length-X[k].shape[0],1))))))
                
        else:
            
            if len(X[k].shape) > 1:
                X_padded.append(np.array(X[k]))
            else:
                X_padded.append(np.array(X[k]).reshape((-1,1)))
  

    X_padded      = np.array(X_padded)

    return X_padded


def flatten_sequences_to_numpy(sequence_list):
    
    seqLists   = [list(itertools.chain.from_iterable(sequence_list[k].tolist())) for k in range(len(sequence_list))]
    flat_seqs  = np.array(list(itertools.chain.from_iterable(seqLists)))
    
    return flat_seqs



class SeqModel:
    
    '''
    Parent class for all RNN models.

    '''
    
    def __init__(self, 
                 maximum_seq_length, 
                 input_dim, 
                 output_dim=1,
                 model_type='RNN',
                 rnn_type='RNN',
                 latent=False,
                 generative=False,
                 irregular=False,
                 multitask=False,
                 prediction_mode='Sequence_labeling',
                 input_name="Input", 
                 output_name="Output",
                 model_name="SeqModel",
                 num_iterations=20, 
                 num_epochs=10, 
                 batch_size=100, 
                 learning_rate=0.0005, 
                 num_rnn_hidden=200, 
                 num_rnn_layers=1,
                 dropout_keep_prob=None,
                 num_out_hidden=200, 
                 num_out_layers=1,
                 **kwargs
                ):
        
        # Set all model variables

        self.maximum_seq_length = maximum_seq_length 
        self.input_dim          = input_dim
        self.output_dim         = output_dim
        self.model_type         = model_type
        self.rnn_type           = rnn_type
        self.latent             = latent
        self.generative         = generative
        self.irregular          = irregular
        self.multitask          = multitask
        self.prediction_mode    = prediction_mode
        self.input_name         = input_name 
        self.output_name        = output_name 
        self.model_name         = model_name
        self.num_iterations     = num_iterations
        self.num_epochs         = num_epochs
        self.batch_size         = batch_size
        self.learning_rate      = learning_rate
        self.num_rnn_hidden     = num_rnn_hidden
        self.num_rnn_layers     = num_rnn_layers
        self.dropout_keep_prob  = dropout_keep_prob
        self.num_out_hidden     = num_out_hidden
        self.num_out_layers     = num_out_layers
        
        
        self.build_rnn_model()
        tf.reset_default_graph()
        self.build_rnn_graph()
        
        
    
    def build_rnn_model(self):
        
        # replace this with dictionary style indexing
        
        model_options_names     = ['RNN','LSTM','GRU','PhasedLSTM']
        
        optimizer_options_names = []
        
        
        model_options   = [BasicRNNCell(self.num_rnn_hidden), rnn_cell.LSTMCell(self.num_rnn_hidden), 
                                   rnn_cell.GRUCell(self.num_rnn_hidden), PhasedLSTMCell(self.num_rnn_hidden)]
        
        self._rnn_model = model_options[np.where(np.array(model_options_names)==self.rnn_type)[0][0]]
        
        if self.dropout_keep_prob is not None:
            
            self._rnn_model = tf.nn.rnn_cell.DropoutWrapper(self._rnn_model, output_keep_prob=self.dropout_keep_prob)
        
        self._Losses = []
        

    def build_rnn_graph(self):
        
        self.data   = tf.placeholder(tf.float32, 
                                     [None, self.maximum_seq_length, self.input_dim], 
                                     name=self.input_name)
            
        self.target = tf.placeholder(tf.float32, 
                                     [None, self.maximum_seq_length, self.output_dim]) 
        
        if self.irregular:
            
            self.times      = tf.placeholder(tf.float32, [None, self.maximum_seq_length, 1])
            self.rnn_input  = (self.times, self.data)
        
        else:
            
            self.rnn_input  = self.data 

            
    @lazy_property
    def length(self):
        
        used   = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        
        return length

    @lazy_property
    def prediction(self):
        
        self.process_rnn_inputs()
        
        # Recurrent network.   
        if self.model_type != 'Seq2SeqAttention': 
            
            rnn_output, _  = rnn.dynamic_rnn(self._rnn_model, 
                                             self.rnn_input_, 
                                             dtype=tf.float32, 
                                             sequence_length=self.length_,)
            
        else:
            
            
            try:
                
                tf.nn.seq2seq = tf.contrib.legacy_seq2seq
                tf.nn.rnn_cell = tf.contrib.rnn
                tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
                print("TensorFlow version : >= 1.0")
            
            except: 
            
                print("TensorFlow version : 0.12")
            
            self.enc_inp    = [self.rnn_input_[:, t, :] for t in range(self.maximum_seq_length)]

            self.dec_output = [tf.placeholder(tf.float32, shape=(None, 1), 
                                              name="dec_output_".format(t)) for t in range(self.maximum_seq_length)]

            self.dec_inp    = [tf.zeros_like(self.enc_inp[0], dtype=np.float32, name="GO")] + self.enc_inp[:-1] 

            self.cells = []
    
            for i in range(self.num_rnn_layers):
                
                with tf.variable_scope('RNN_{}'.format(i)):
                    
                    self.cells.append(tf.nn.rnn_cell.GRUCell(self.num_rnn_hidden))
            
            
            cell  = tf.nn.rnn_cell.MultiRNNCell(self.cells)
            self.dec_outputs, self.dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(self.enc_inp, self.dec_inp, cell) 
            
            self.weight_dec, self.bias_dec = self._weight_and_bias(self.num_rnn_hidden, 1, ["w_dec", "b_dec"])
            
            self.seq2seq_attn = [(tf.matmul(i, self.weight_dec) + self.bias_dec) for i in self.dec_outputs]
            self.seq2seq_attn = tf.nn.softmax(tf.reshape(tf.stack(self.seq2seq_attn), 
                                                         [-1, self.maximum_seq_length, 1]), axis=1)
            
        
        # Softmax layer.
        self.weight_0, self.bias_0 = self._weight_and_bias(self.input_dim, 
                                                           self.num_out_hidden, 
                                                           ["w_0", "b_0"])
        
        self.weight, self.bias     = self._weight_and_bias(self.num_out_hidden, 
                                                           self.output_dim, 
                                                           ["w", "b"])
            
        # Flatten to apply same weights to all time steps.
        
        if self.model_type not in ['RETAIN', 'Seq2SeqAttention']: 
            
            rnn_output  = tf.reshape(rnn_output, [-1, self.num_out_hidden])
            
            prediction  = tf.nn.sigmoid(tf.matmul(rnn_output, self.weight) + self.bias)
        
        elif self.model_type == 'RETAIN':
            
            self.weight_a, self.bias_a = self._weight_and_bias(self.num_out_hidden, 
                                                               self.output_dim, 
                                                               ["w_a", "b_a"])
            
            rnn_output      = tf.reshape(rnn_output, [-1, self.num_out_hidden])
            
            self.attention  = tf.nn.softmax(tf.reshape(tf.matmul(rnn_output, self.weight_a) + self.bias_a, 
                                                       [-1, self.maximum_seq_length, 1]), axis=1)
            
            attn_mask       = tf.expand_dims(tf.sign(tf.reduce_max(tf.abs(self.rnn_input_), reduction_indices=2)), axis=2)
            masked_attn     = tf.multiply(attn_mask, self.attention)
            attn_norms      = tf.expand_dims(tf.tile(tf.reduce_sum(masked_attn, axis=1), [1, self.maximum_seq_length]), axis=2)
            self.attention  = masked_attn/attn_norms
            self.attention_ = tf.tile(self.attention, [1, 1, self.input_dim])
            self.context    = tf.reduce_sum(tf.multiply(self.attention_, self.rnn_input_), reduction_indices=1)
            context_layer   = tf.matmul(self.context, self.weight_0) + self.bias_0
            prediction      = tf.nn.sigmoid(tf.matmul(context_layer, self.weight) + self.bias)
        
        elif self.model_type == 'Seq2SeqAttention':
            
            self.attention  = self.seq2seq_attn
            
            attn_mask       = tf.expand_dims(tf.sign(tf.reduce_max(tf.abs(self.rnn_input_), reduction_indices=2)), axis=2)
            masked_attn     = tf.multiply(attn_mask, self.attention)
            attn_norms      = tf.expand_dims(tf.tile(tf.reduce_sum(masked_attn, axis=1), [1, self.maximum_seq_length]), axis=2)
            self.attention  = masked_attn/attn_norms
            self.attention_ = tf.tile(self.attention, [1, 1, self.input_dim])
            self.context    = tf.reduce_sum(tf.multiply(self.attention_, self.rnn_input_), reduction_indices=1)
            context_layer   = tf.matmul(self.context, self.weight_0) + self.bias_0
            prediction      = tf.nn.sigmoid(tf.matmul(context_layer, self.weight) + self.bias)

        prediction      = tf.reshape(prediction, [-1, self.maximum_seq_length, self.output_dim])
        self.predicted  = prediction
        self.predicted  = tf.identity(self.predicted, name=self.output_name)
        
        return prediction

    
    def process_rnn_inputs(self):
        
        if self.model_type in ['RETAIN', 'Seq2SeqAttention']: 
            
            self.num_samples = tf.shape(self.data)[0]
             
            Lengths_         = np.repeat(self.length, self.maximum_seq_length)
            
            conv_data        = tf.reshape(tf.tile(self.data, [1, self.maximum_seq_length, 1]), 
                                          [self.maximum_seq_length * self.num_samples, 
                                           self.maximum_seq_length, self.input_dim])
            
            conv_mask_       = tf.ones([self.maximum_seq_length, self.maximum_seq_length], tf.float32)
            
            conv_mask        = tf.tile(tf.expand_dims(tf.tile(tf.matrix_band_part(conv_mask_, -1, 0), 
                                                              [self.num_samples, 1]), 2), 
                                                              [1, 1, self.input_dim])
            
            masked_data   = tf.multiply(conv_data, conv_mask)
            
            Seq_lengths_  = tf.tile(tf.range(1, self.maximum_seq_length + 1, 1), [self.num_samples])
            
            if self.model_type == 'RETAIN':
            
                self.rnn_input_  = tf.reverse_sequence(masked_data, batch_axis=0, seq_dim=1, 
                                                       seq_lengths=Seq_lengths_, seq_axis=None)
            else:
                
                self.rnn_input_  = masked_data
                
            
            used         = tf.sign(tf.reduce_max(tf.abs(self.rnn_input_), reduction_indices=2))
            length       = tf.reduce_sum(used, reduction_indices=1)
            self.length_ = tf.cast(length, tf.int32)
            
            self.target_ = tf.tile(self.target, [self.maximum_seq_length, 1, 1]) 
            
        else:    
            
            self.rnn_input_ = self.rnn_input
            self.target_    = self.target 
            self.length_    = self.length

    @lazy_property
    def loss(self):
        
        # Compute cross entropy for each frame.
        cross_entropy  = tf.reduce_sum(-1*(self.target * tf.log(self.prediction) + (1-self.target)*(tf.log(1-self.prediction))),
                                       reduction_indices=2) 

        mask           = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        cross_entropy *= mask
        self.mask      = mask
        
        # Average over actual sequence lengths.
        cross_entropy  = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        
        return tf.reduce_mean(cross_entropy)
        
    
    
    def train(self, X, Y, T=None):
        
        X_, Y_   = padd_data(X, self.maximum_seq_length), padd_data(Y, self.maximum_seq_length)
        
        if T is not None:
            T_   = padd_data(T, self.maximum_seq_length)
        
        
        sess = tf.InteractiveSession()
        
        opt      = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        init     = tf.global_variables_initializer()
        
        sess.run(init)

        saver = tf.train.Saver()
        
        for epoch in range(self.num_epochs):
                
            for _ in range(self.num_iterations):
                
                batch_samples = np.random.choice(list(range(X_.shape[0])), size=self.batch_size, replace=False)
                batch_train   = X_[batch_samples,:,:]
                batch_targets = Y_[batch_samples,:,:] 
                
                if T is not None:
                    batch_times = T_[batch_samples,:,:]
                    
                    train_dict  = {self.data   : batch_train,
                                   self.target : batch_targets,
                                   self.times  : batch_times}
                else:
                    
                    train_dict  = {self.data   : batch_train,
                                   self.target : batch_targets}
                
                
                sess.run(opt, feed_dict=train_dict)
                
                Loss          = sess.run(self.loss, feed_dict=train_dict)
                
                self._Losses.append(Loss)
 
                # Visualize function
                print('Epoch {} \t----- \tBatch {} \t----- \tLoss {}'.format(epoch, _, self._Losses[-1]))
  
        # change names
        saver.save(sess, "./mlaimRNN_model") 
        
        tf.saved_model.simple_save(sess, export_dir='modelgraph', inputs={"myInput": self.data}, 
                                   outputs={"myOutput": self.predicted})   
            
    def predict(self, X, T=None):
        
        with tf.Session() as sess:
            
            saver           = tf.train.import_meta_graph("mlaimRNN_model.meta")
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            
            preds_lengths   = [len(X[k]) for k in range(len(X))]
            
            X_pred          = padd_data(X, padd_length=self.maximum_seq_length)
            
            if T is not None:
                T_pred      = padd_data_enforce(T, padd_length=self.maximum_seq_length)
                pred_dict   = {self.data   : X_pred, self.times   : T_pred}
            else:
                pred_dict   = {self.data   : X_pred}
            
            prediction_     = sess.run(self.prediction, pred_dict).reshape([-1, self.maximum_seq_length, 1])         

            preds_          = []
            
            for k in range(len(X)):
                
                preds_.append(prediction_[k, 0 : preds_lengths[k]])
                
                
            if self.model_type in ['RETAIN', 'Seq2SeqAttention']: 
                
                attn_                  = sess.run(self.attention, pred_dict) 
                attn_per_patient       = [attn_[u * self.maximum_seq_length : u * self.maximum_seq_length + self.maximum_seq_length, :, :] for u in range(len(X))]
                attn_lists_per_patient = [[attn_per_patient[u][k, 0 : k + 1, :] for k in range(self.maximum_seq_length)] for u in range(len(X))]
                
                preds_                 = (preds_, attn_lists_per_patient)
            
        return preds_    
    
    
    def evaluate(self, preds, Y_test):
        
        flat_preds   = flatten_sequences_to_numpy(preds)
        flat_Y_test  = np.array(list(itertools.chain.from_iterable([Y_test[k].tolist() for k in range(len(Y_test))])))
        
        _performance = roc_auc_score(flat_Y_test, flat_preds)
        
        return _performance
    
    @staticmethod
    def _weight_and_bias(in_size, out_size, wnames):
    
        weight = tf.get_variable(wnames[0], shape=[in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        bias   = tf.get_variable(wnames[1], shape=[out_size], initializer=tf.contrib.layers.xavier_initializer())
        
        return weight, bias
        


