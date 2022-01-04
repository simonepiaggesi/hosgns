# Copyright (C) 2020 Simone Piaggesi (Python3 script),
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <http://www.gnu.org/licenses/>.
#

import tensorflow as tf
from tensorflow.keras import Model
# import node2vec
from snap_node2vec import snap_node2vec
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import time
import itertools as it

from sklearn.utils.extmath import cartesian
from sklearn.utils.random import sample_without_replacement
from random import shuffle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class TFEmbedder(Model):
    '''
    Tensorflow object for trainable embedding matrices W, C, T, ...
    '''
    def __init__(self, init_func, vsizes, embdim):
        super(TFEmbedder, self).__init__()

        self.vsizes = vsizes
        self.n_factors = len(vsizes)
        self.embdim = embdim
        self.factors = []
        
        for i in range(self.n_factors):
            self.factors.append(tf.Variable(init_func(vsizes[i], embdim), trainable=True, name="factor_%d"%i))

    def call(self, inputs):

        Mhat_batch = tf.gather(self.factors[0], inputs[0], axis=0)
        for i in range(1, self.n_factors):
            Mhat_batch = Mhat_batch * tf.gather(self.factors[i], inputs[i], axis=0)

        return tf.reduce_sum(Mhat_batch, axis=1)


class TFEmbedderWeightTying(TFEmbedder):
    '''
    Tensorflow object for trainable embedding matrices W, C, T, ... when we allow weight tying among different factors
    '''
    def __init__(self, init_func, vsizes, fshares, embdim):
        super(TFEmbedderWeightTying, self).__init__(init_func, vsizes, embdim)
        self.fshares = fshares

    def call(self, inputs):

        Mhat_batch = tf.gather(self.factors[self.fshares[0]], inputs[0], axis=0)
        for i in range(1, len(self.fshares)):
            Mhat_batch = Mhat_batch * tf.gather(self.factors[self.fshares[i]], inputs[i], axis=0)

        return tf.reduce_sum(Mhat_batch, axis=1)

class HOSGNSSampler:
    '''
    Object for HOSGNS sampling
    '''
    
    def __init__(self, network, window_size, batch_size, negative_samples, random_state, walk_len=80, num_walks=10):
        
        self.supra_G = network
        self.window_size = window_size
        self.walk_len = walk_len
        self.num_walks = num_walks
        
        self.samples_per_sentence = self.window_size*(2*self.walk_len-1)-self.window_size*self.window_size
        self.batch_size = int(batch_size/self.samples_per_sentence)*self.samples_per_sentence # number of positive examples sampled
        
        self.k_neg = negative_samples
        self.random_state = random_state
                
        node_name = list(self.supra_G.nodes())
        node_index = {node:index for index, node in enumerate(node_name)}
        supra_H = nx.relabel_nodes(self.supra_G, node_index)
        
        unique_node_name = np.unique([int(n_node.split('-')[0]) for n_node in node_name])
        unique_time_name = np.unique([int(n_node.split('-')[1]) for n_node in node_name])
        self.nnodes = unique_node_name.shape[0]
        self.ntimes = unique_time_name.shape[0]
        map_node_index = {node:index for index, node in enumerate(unique_node_name)}
        map_time_index = {time:index for index, time in enumerate(unique_time_name)}

        self.map_to_node = np.array([map_node_index[int(n_node.split('-')[0])] for n_node in node_name])
        self.map_to_time = np.array([map_time_index[int(n_node.split('-')[1])] for n_node in node_name])
        
        print('Sampling Random Walks...')
        node2vec = snap_node2vec(d=2, max_iter=1, walk_len=walk_len, num_walks=num_walks, con_size=10, ret_p=1, inout_p=1)
        self.corpus, _ = node2vec.sample_random_walks(supra_H, edge_f = None, is_weighted = True, 
                                                 no_python=True, directed=supra_H.is_directed())
        self.flat_corpus = np.array([item for sublist in self.corpus for item in sublist]).astype(int)
        shuffle(self.corpus)
                
        def samples_generator_fn(size):
            eff_size = int(size/self.samples_per_sentence)
            cycles = int(len(self.corpus)/eff_size)
            while True:
                for i in range(cycles):
                    yield self.corpus[i*eff_size:(i+1)*eff_size]
                shuffle(self.corpus)
                            
        self.samples_generator = samples_generator_fn(size=self.batch_size)
        
    def __call__(self):
        
        random_state = self.random_state
        sentences =  next(self.samples_generator) 

        training_data = []
        # Cycle through each sentence in corpus
        for sentence in sentences:
            sent_len = len(sentence)
            # Cycle through each word in sentence
            for i, word in enumerate(sentence):
                # Convert target word to one-hot
                w_target = int(sentence[i])
                #node_target = self.map_to_node[w_target]
                #time_target = self.map_to_time[w_target]
                
                # Cycle through context window
                # Note: window_size 2 will have range of 5 values
                for j in range(i-self.window_size, i+self.window_size+1):
                    # Criteria for context word 
                    # 1. Target word cannot be context word (j != i)
                    # 2. Index must be greater or equal than 0 (j >= 0) - if not list index out of range
                    # 3. Index must be less or equal than length of sentence (j <= sent_len-1) - if not list index out of range 
                    if j != i and j <= sent_len-1 and j >= 0:
                        # Append the one-hot representation of word to w_context
                        w_context = int(sentence[j])
                        #node_context = self.map_to_node[w_context]
                        #time_context = self.map_to_time[w_context]
                         
                        # training_data contains a one-hot representation of the target word and context words
                        training_data.append([w_target, w_context])
                        
        positive = np.array(training_data)
        
        positive = np.concatenate([self.map_to_node[positive[:,[0]]], self.map_to_time[positive[:,[0]]], \
                    self.map_to_node[positive[:,[1]]], self.map_to_time[positive[:,[1]]]], axis=1)

        negative = np.concatenate(
                [random_state.choice(self.flat_corpus, size=(self.batch_size*self.k_neg, 1))
                 for i in range(3)], axis=1)
        
        negative = np.concatenate([np.tile(positive[:,0], self.k_neg)[:,np.newaxis], self.map_to_time[negative[:,[0]]], \
                    self.map_to_node[negative[:,[1]]], self.map_to_time[negative[:,[2]]]], axis=1)
        
        return positive, negative
    
class HOSGNSSolver:
    '''
    Tensorflow object the HOSGNS optimization
    '''
    def __init__(self, network, window_size, order, emb_dim, iters, batch_size, negative_samples,
                 learning_rate, warmup_steps=0, weight_tying=False, random_state=None):
        
        self.d = emb_dim # embedding dimension
        self.n_iters = iters # number of training iterations
        self.warmup_steps = warmup_steps
        self.random_state = random_state
        
        self.k_neg = negative_samples # negative sampling constant
        self.lr = learning_rate
        self.order = order
        
        self.wt = weight_tying # boolean variable to make weight tying
            
        if not isinstance(self.random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(self.random_state)
        tf.random.set_seed(self.random_state.get_state()[1][0])
        
        self.sampler = HOSGNSSampler(network=network, window_size=window_size,
                                     batch_size=batch_size, negative_samples=self.k_neg, random_state=self.random_state)
        
        self.batch_size = self.sampler.batch_size # number of positive examples sampled
        
        self.vsizes = [self.sampler.nnodes, self.sampler.ntimes, self.sampler.nnodes, self.sampler.ntimes] # sizes of each axis
        self.vsizes = self.vsizes[:self.order] 
        
        # make the inititalization function for the embeddings (normal distribution)
        nwi = lambda v,e: tf.random.normal((v,e), 0.0, 1.0/e)
        # define the model
        if not self.wt:
            self.model = TFEmbedder(nwi, self.vsizes, self.d)
        else:
            self.model = TFEmbedderWeightTying(nwi, self.vsizes, [0,1,0,1], self.d)

    def train(self, print_loss='sg', print_every=1):
        '''
        Train function
        --------------
        Parameters

        print_loss : 'sg' (skip-gram loss) , 'tf' (tensor factorization loss), 'sg-tf' (both)
        print_every : steps frequency for printing and saving losses

        Returns
        -------

        losses : dictionary of losses values every 'print_every' steps
        '''
    
        random_state = self.random_state
        print_sg = 'sg' in print_loss.split('-')

        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        
        def make_warmup_step():
            warmup_lr = tf.keras.optimizers.schedules.PolynomialDecay(0.05, self.warmup_steps)
            optimizer = tf.keras.optimizers.Adam(learning_rate=warmup_lr)

            @tf.function
            def warmup_minibatch(scope_model, batch):
                with tf.GradientTape() as tape:
                    y_pred = scope_model(batch)[:, tf.newaxis]
                    loss = (y_pred + np.log(self.k_neg))**2
                gradients = tape.gradient(loss, scope_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, scope_model.trainable_variables))

            return warmup_minibatch
        
        warmup_minibatch = make_warmup_step()

        def make_train_step():
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
            train_loss = tf.keras.metrics.Mean(name='train_loss')

            @tf.function
            def train_minibatch(scope_model, batch, y_true):
                with tf.GradientTape() as tape:
                    y_pred = tf.math.sigmoid(scope_model(batch))[:, tf.newaxis]
                    loss = bce(y_true, y_pred)
                gradients = tape.gradient(loss, scope_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, scope_model.trainable_variables))
                train_loss(loss)

            return train_loss, train_minibatch

        # Get the special objects we will be using for training
        train_loss, train_minibatch = make_train_step()

        start = time.time()
        estart = time.time()

        losses = {'tf':[], 'sg':[]}
        
        if self.warmup_steps > 0:
            print('Warmup...')
            for i in range(self.warmup_steps):
                samples = np.concatenate(
                    [random_state.choice(self.vsizes[i], size=(self.batch_size*2, 1))
                     for i in range(self.order)], axis=1)
                batch_tuple = tf.tuple([samples[:,i] for i in range(self.order)])
                warmup_minibatch(self.model, batch_tuple)

        batch_labels = tf.concat([tf.ones((self.batch_size,1)), tf.zeros((self.batch_size*self.k_neg,1))], axis=0)
        #batch_weights = tf.concat([tf.ones((self.batch_size,)), tf.ones((self.batch_size,))*self.k_neg], axis=0)
        
        print('Training...')
        for i in range(self.n_iters+1):

            pos_samples, neg_samples = self.sampler()

            pos_neg = np.concatenate((pos_samples[:,:self.order], neg_samples[:,:self.order]), axis=0)
            batch_tuple = tf.tuple([pos_neg[:,i] for i in range(self.order)])

            train_minibatch(self.model, batch_tuple, batch_labels)

            if i % print_every==0:
                if not print_loss==False:
                    if print_sg:
                        current_loss = train_loss.result()
                        losses['sg'].append(current_loss.numpy())
                        print('step {:4} - loss: {} ({:0.4f} seconds)'.format(
                                    i, current_loss.numpy(), time.time() - estart))
                    estart = time.time()

        print('\nTotal time: {:0.4f} seconds.'.format(time.time() - start))
        return losses
