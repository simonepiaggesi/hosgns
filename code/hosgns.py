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
import numpy as np
import pandas as pd
import scipy
import time
import itertools as it
from utils import sigmoid
from sklearn.utils.extmath import cartesian
from sklearn.utils.random import sample_without_replacement

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
            self.factors.append(tf.Variable(init_func(vsizes[i], embdim), name="factor_%d"%i))

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


class HOSGNSSolver:
    '''
    Tensorflow object the HOSGNS optimization
    '''
    def __init__(self, tensor, marginals, active_events_list,
                 emb_dim,iters, batch_size, negative_samples,
                 learning_rate, weight_tying=False):
        self.tensor = tensor # probabilities of positivie examples (higher order tensor reshaped to a dense or csr matrix)
        self.d = emb_dim # embedding dimension
        self.active_ijk = active_events_list # list of active elements of axis 0
        self.n_iters = iters # number of training iterations
        self.batch_size = batch_size # number of examples sampled (positive and negative)
        self.k_neg = negative_samples # negative sampling constant
        self.lr = learning_rate
        self.marginals = marginals # tuple of marginal probabilities for negative examples
        self.order = len(marginals)
        self.vsizes = [m.shape[0] for m in marginals # sizes of each axis

        self.wt = weight_tying # boolean variable to make weight tying
        self.sp = scipy.sparse.issparse(self.tensor)

        if self.sp:
            self.nijk = self.tensor[self.active_ijk, :].sum(axis=1).A.ravel()
        else:
            self.nijk = self.tensor[self.active_ijk, :].sum(axis=1)

        # make the inititalization function for the embeddings (normal distribution)
        nwi = lambda v,e: tf.random.normal((v,e), 0.0, 1.0/e)
        # define the model
        if not self.wt:
            self.model = TFEmbedder(nwi, self.vsizes, self.d)
        else:
            self.model = TFEmbedderWeightTying(nwi, self.vsizes, [0,1,0,1], self.d)

    def tensor_factorization_loss(self):
        # function to compute and monitor the tensor factorization loss (not directly minimized by the model)
        samples = [
            np.sort(sample_without_replacement(n_population=vsize, n_samples=32))
            for vsize in self.vsizes
        ]

        ijk_ = self.all_tuples_indices[self.all_tuples.isin(it.product(*samples[:-1]))]

        num = self.tensor[ijk_][:, samples[-1]]
        if self.sp:
            num = num.toarray()

        den = self.marginals[0][samples[0]]
        for i in range(1, self.order):
            den = np.tensordot(den, self.marginals[i][samples[i]], axes=0)
        PMI_np = np.log(num.ravel()/den.ravel())

        batch = cartesian(samples)
        return  np.sum(np.abs(sigmoid(PMI_np - np.log(self.k_neg))\
            - sigmoid(self.model(tf.tuple([batch[:,i] for i in range(self.order)])))))

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

        print_tf = 'tf' in print_loss.split('-')
        print_sg = 'sg' in print_loss.split('-')

        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

        def make_train_step():
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
            train_loss = tf.keras.metrics.Mean(name='train_loss')

            @tf.function
            def train_minibatch(scope_model, batch, y_true, sample_weight):
                with tf.GradientTape() as tape:
                    y_pred = tf.math.sigmoid(scope_model(batch))[:, tf.newaxis]
                    loss = bce(y_true, y_pred, sample_weight)/self.batch_size
                gradients = tape.gradient(loss, scope_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, scope_model.trainable_variables))
                train_loss(loss)

            return train_loss, train_minibatch

        # Get the special objects we will be using for training
        train_loss, train_minibatch = make_train_step()

        tensor = self.tensor[self.active_ijk, :]

        # indices
        nr_active = self.nijk.shape[0]

        idxs = [np.arange(vsize, dtype=np.int32) for vsize in self.vsizes]

        all_tuples = cartesian(idxs[:-1])
        active_tuples = all_tuples[self.active_ijk]

        if print_tf:
            self.all_tuples = pd.Series(list(map(tuple, all_tuples)))
            self.all_tuples_indices = np.arange(np.product(self.vsizes[:-1]), dtype=np.int32)

        start = time.time()
        estart = time.time()

        losses = {'tf':[], 'sg':[]}

        batch_labels = tf.concat([tf.ones((self.batch_size,1)), tf.zeros((self.batch_size,1))], axis=0)
        batch_weights = tf.concat([tf.ones((self.batch_size,)), tf.ones((self.batch_size,))*self.k_neg], axis=0)

        for i in range(self.n_iters+1):

            active_events =  np.random.choice(nr_active, size=self.batch_size, p=self.nijk)

            #sparse cumsum
            if self.sp:
                cs_l_ijk = (tensor[active_events].toarray() / self.nijk[active_events, np.newaxis]).cumsum(axis=1)
                active_cotimes = (cs_l_ijk >
                                  np.random.rand(self.batch_size)[:,np.newaxis]).argmax(axis=1)
            #dense cumsum
            else:
                cs_l_ijk = (tensor / self.nijk[:, np.newaxis]).cumsum(axis=1)
                active_cotimes = (cs_l_ijk[active_events] >
                                  np.random.rand(self.batch_size)[:,np.newaxis]).argmax(axis=1)

            pos_samples = np.concatenate((active_tuples[active_events], active_cotimes[:,np.newaxis]), axis=1)

            neg_samples = np.concatenate(
                [np.random.choice(self.vsizes[i], size=(self.batch_size, 1), p=self.marginals[i])
                 for i in range(self.order)], axis=1)

            pos_neg = np.concatenate((pos_samples, neg_samples), axis=0)
            batch_tuple = tf.tuple([pos_neg[:,i] for i in range(self.order)])

            train_minibatch(self.model, batch_tuple, batch_labels, batch_weights)

            if i % print_every==0:
                if not print_loss==False:
                    if print_sg:
                        current_loss = train_loss.result()
                        losses['sg'].append(current_loss.numpy())
                        print('step {:4} - loss: {} ({:0.4f} seconds)'.format(
                                    i, current_loss.numpy(), time.time() - estart))
                    if print_tf:
                        losses['tf'].append(self.tensor_factorization_loss())
                    estart = time.time()


        print('\nTotal time: {:0.4f} seconds.'.format(time.time() - start))
        return losses
