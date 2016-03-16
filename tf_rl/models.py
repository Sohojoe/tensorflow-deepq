import math
import tensorflow as tf
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, RMSprop, Adam
from keras.regularizers import l2
from keras import backend as K
from keras.utils import np_utils

from .utils import base_name

import numpy as np

class Layer(object):
    def __init__(self, input_sizes, output_size, scope):
        """Cretes a neural network layer."""
        if type(input_sizes) != list:
            input_sizes = [input_sizes]

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.scope       = scope or "Layer"

        with tf.variable_scope(self.scope):
            self.Ws = []
            for input_idx, input_size in enumerate(input_sizes):
                W_name = "W_%d" % (input_idx,)
                W_initializer =  tf.random_uniform_initializer(
                        -1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size))
                W_var = tf.get_variable(W_name, (input_size, output_size), initializer=W_initializer)
                self.Ws.append(W_var)
            self.b = tf.get_variable("b", (output_size,), initializer=tf.constant_initializer(0))

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        assert len(xs) == len(self.Ws), \
                "Expected %d input vectors, got %d" % (len(self.Ws), len(xs))
        with tf.variable_scope(self.scope):
            return sum([tf.matmul(x, W) for x, W in zip(xs, self.Ws)]) + self.b

    def variables(self):
        return [self.b] + self.Ws

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"

        with tf.variable_scope(scope) as sc:
            for v in self.variables():
                tf.get_variable(base_name(v), v.get_shape(),
                        initializer=lambda x,dtype=tf.float32: v.initialized_value())
            sc.reuse_variables()
            return Layer(self.input_sizes, self.output_size, scope=sc)


class MLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities, scope=None, given_layers=None):
        self.input_sizes = input_sizes
        self.hiddens = hiddens
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        self.scope = scope or "MLP"

        assert len(hiddens) == len(nonlinearities), \
                "Number of hiddens must be equal to number of nonlinearities"

        with tf.variable_scope(self.scope):
            if given_layers is not None:
                self.input_layer = given_layers[0]
                self.layers      = given_layers[1:]
            else:
                self.input_layer = Layer(input_sizes, hiddens[0], scope="input_layer")
                self.layers = []

                for l_idx, (h_from, h_to) in enumerate(zip(hiddens[:-1], hiddens[1:])):
                    self.layers.append(Layer(h_from, h_to, scope="hidden_layer_%d" % (l_idx,)))

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        with tf.variable_scope(self.scope):
            hidden = self.input_nonlinearity(self.input_layer(xs))
            for layer, nonlinearity in zip(self.layers, self.layer_nonlinearities):
                hidden = nonlinearity(layer(hidden))
            return hidden

    def variables(self):
        res = self.input_layer.variables()
        for layer in self.layers:
            res.extend(layer.variables())
        return res

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        nonlinearities = [self.input_nonlinearity] + self.layer_nonlinearities
        given_layers = [self.input_layer.copy()] + [layer.copy() for layer in self.layers]
        return MLP(self.input_sizes, self.hiddens, nonlinearities, scope=scope,
                given_layers=given_layers)


def custom_initialization(shape, scale=0.003, name=None):
    return K.variable(np.random.uniform(low=-scale, high=scale, size=shape), name=name)


class KERASMLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities, scope=None, given_layers=None, weights=None, regularizer=False):
        self.input_sizes = input_sizes

        # if type(self.input_sizes) != list:
        #     self.input_sizes = [self.input_sizes]

        self.hiddens = hiddens
        self.nonlinearities = nonlinearities
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        self.scope = scope or "MLP"
        self.input_weights = weights
        self.use_regularization = regularizer

        assert len(hiddens) == len(nonlinearities), \
                "Number of hiddens must be equal to number of nonlinearities"

        self.model = Sequential()

        #self.model.add(BatchNormalization(input_shape=[self.input_sizes]))
        #self.model.add(Dense(hiddens[0], init="lecun_uniform"))
        if regularizer:
            self.model.add(Dense(hiddens[0], input_dim=input_sizes, init="lecun_uniform", W_regularizer=l2(0.01)))
        else:
            self.model.add(Dense(hiddens[0], input_dim=input_sizes, init="lecun_uniform"))
        self.model.add(Activation(self.input_nonlinearity))
        #self.model.add(BatchNormalization())
        for l_idx, (h_from, h_to) in enumerate(zip(hiddens[:-2], hiddens[1:-1])):
            if regularizer:
                self.model.add(Dense(h_to, init='lecun_uniform', W_regularizer=l2(0.01)))
            else:
                self.model.add(Dense(h_to, init='lecun_uniform'))

            self.model.add(Activation(self.layer_nonlinearities[l_idx]))
            #self.model.add(BatchNormalization())


        if regularizer:
            self.model.add(Dense(hiddens[-1], init=custom_initialization, W_regularizer=l2(0.01)))
        else:
            self.model.add(Dense(hiddens[-1], init=custom_initialization))
        self.model.add(Activation(self.layer_nonlinearities[-1]))

        lr = 0.0001
        if regularizer:
            lr = 0.001
        self.opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='MSE', optimizer=self.opt)
        if weights is not None:
            self.model.set_weights(weights)

    def __call__(self, xs):
        if len(xs) > 1:
            xs = np.matrix(xs)
        xs[0,0] = xs[0,0] / np.pi
        xs[0,1] = xs[0,1] / (2.0 * np.pi)
        # xs[0,2] = xs[0,2] / np.pi
        # xs[0,3] = xs[0,3] / 20.0
        return self.model.predict([xs], batch_size=len(xs))

    def variables(self):
        return self.model.get_weights()

    def save(self, filepath):
        self.model.save_weights(filepath, overwrite=False)

    def restore(self, filepath):
        self.model.load_weights(filepath)

    def copy(self, scope=None):
        # print '---------------------'
        # print len(self.model.get_weights())
        # print '---------------------'
        # for i, x in enumerate(self.model.get_weights()):
        #     print len(self.model.get_weights()[i])
        #     # print x
        #     if not isinstance(x[0], np.float32):
        #         print '-- ' + str(len(x[0]))
        #     print ' '
        # print '#####################'

        return KERASMLP(self.input_sizes, self.hiddens, self.nonlinearities, scope=scope, weights=self.model.get_weights(), regularizer=self.use_regularization)


class PolicyMLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities, scope=None, given_layers=None, weights=None, regularizer=False):
        self.input_sizes = input_sizes

        # if type(self.input_sizes) != list:
        #     self.input_sizes = [self.input_sizes]

        self.hiddens = hiddens
        self.nonlinearities = nonlinearities
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        self.scope = scope or "MLP"
        self.input_weights = weights
        self.use_regularization = regularizer

        assert len(hiddens) == len(nonlinearities), \
                "Number of hiddens must be equal to number of nonlinearities"

        self.model = Sequential()
        if regularizer:
            self.model.add(Dense(hiddens[0], input_dim=input_sizes, init="lecun_uniform", W_regularizer=l2(0.01)))
        else:
            self.model.add(Dense(hiddens[0], input_dim=input_sizes, init="lecun_uniform"))
        self.model.add(LeakyReLU(alpha=0.1))
        for l_idx, (h_from, h_to) in enumerate(zip(hiddens[:-2], hiddens[1:-1])):
            if regularizer:
                self.model.add(Dense(h_to, init='lecun_uniform', W_regularizer=l2(0.01)))
            else:
                self.model.add(Dense(h_to, init='lecun_uniform'))

            self.model.add(LeakyReLU(alpha=0.1))

        if regularizer:
            self.model.add(Dense(hiddens[-1], init=custom_initialization, W_regularizer=l2(0.01)))
        else:
            self.model.add(Dense(hiddens[-1], init=custom_initialization))
        self.model.add(Activation('tanh'))

        lr = 0.01
        self.opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='MSE', optimizer=self.opt)
        if weights is not None:
            self.model.set_weights(weights)

    def __call__(self, xs):
        if len(xs) > 1:
            xs = np.matrix(xs)
        xs[0,0] = xs[0,0] / np.pi
        xs[0,1] = 2.0/(1.0 + math.exp(-0.25*xs[0,1])) - 1.0
        # xs[0,2] = xs[0,2] / np.pi
        # xs[0,3] = xs[0,3] / 20.0
        return self.model.predict([xs], batch_size=len(xs))

    def variables(self):
        return self.model.get_weights()

    def save(self, filepath):
        self.model.save_weights(filepath, overwrite=False)

    def restore(self, filepath):
        self.model.load_weights(filepath)

    def copy(self, scope=None):
        return PolicyMLP(self.input_sizes, self.hiddens, self.nonlinearities, scope=scope, weights=self.model.get_weights(), regularizer=self.use_regularization)


class ValueMLP(object):
    def __init__(self, state_sizes, action_sizes, hiddens, nonlinearities, scope=None, given_layers=None, weights=None, regularizer=False):
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes

        # if type(self.input_sizes) != list:
        #     self.input_sizes = [self.input_sizes]

        self.hiddens = hiddens
        self.nonlinearities = nonlinearities
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        self.scope = scope or "MLP"
        self.input_weights = weights
        self.use_regularization = regularizer

        assert len(hiddens) == len(nonlinearities), \
                "Number of hiddens must be equal to number of nonlinearities"

        # Create Graph for full model
        self.model = Graph()
        self.model.add_input('state', input_shape=(self.state_sizes,))
        self.model.add_input('action', input_shape=(self.action_sizes,))

        # Create Sequential container for state processing
        self.state_proc = Sequential()
        if regularizer:
            # self.state_proc.add(BatchNormalization(input_shape=[self.state_sizes]))
            self.state_proc.add(Dense(hiddens[0], input_dim=self.state_sizes, init="lecun_uniform", W_regularizer=l2(0.01)))
            self.state_proc.add(LeakyReLU(alpha=0.1))
        else:
            self.state_proc.add(Dense(hiddens[0], input_dim=self.state_sizes, init="lecun_uniform"))
            self.state_proc.add(LeakyReLU(alpha=0.1))

        # Add State Processer to Graph
        self.model.add_node(self.state_proc, name='state_proc', input='state')

        # Add Dense Layer to combine state processor with action input
        if regularizer:
            self.model.add_node(Dense(hiddens[1], init='lecun_uniform', W_regularizer=l2(0.01)), name='joiner', inputs=['state_proc', 'action'])
        else:
            self.model.add_node(Dense(hiddens[1], init='lecun_uniform'), name='joiner', inputs=['state_proc', 'action'])
        self.model.add_node(LeakyReLU(alpha=0.1), name='activation_0', input='joiner')

        if len(self.hiddens) > 3:
            # Add remaining hidden layers
            last_idx = 0
            for l_idx, (h_from, h_to) in enumerate(zip(hiddens[1:-2], hiddens[2:-1])):
                if regularizer:
                    self.model.add_node(Dense(h_to, init='lecun_uniform', W_regularizer=l2(0.01)), name='dense_%d'%(l_idx+1), input='activation_%d'%(l_idx))
                else:
                    self.model.add_node(Dense(h_to, init='lecun_uniform'), name='dense_%d'%(l_idx+1), input='activation_%d'%(l_idx))
                self.model.add_node(LeakyReLU(alpha=0.1), name='activation_%d'%(l_idx+1), input='dense_%d'%(l_idx+1))
                last_idx = l_idx

            if regularizer:
                self.model.add_node(Dense(hiddens[-1], init=custom_initialization, W_regularizer=l2(0.01)), name='dense_out', input='activation_%d'%(last_idx+1))
            else:
                self.model.add_node(Dense(hiddens[-1], init=custom_initialization), name='dense_out', input='activation_%d'%(last_idx+1))
            self.model.add_node(LeakyReLU(alpha=0.1), name='activation_out', input='dense_out')
        else:
            if regularizer:
                self.model.add_node(Dense(hiddens[-1], init=custom_initialization, W_regularizer=l2(0.01)), name='dense_out', input='activation_0')
            else:
                self.model.add_node(Dense(hiddens[-1], init=custom_initialization), name='dense_out', input='activation_0')
            self.model.add_node(Activation('linear'), name='activation_out', input='dense_out')

        self.model.add_output(name='value_output', input='activation_out')

        lr = 0.01
        self.opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss={'value_output': 'MSE'}, optimizer=self.opt)
        if weights is not None:
            self.model.set_weights(weights)

    def __call__(self, xs, xa):
        if len(xs) > 1:
            xs = np.matrix(xs)
        if len(xa) > 1:
            xa = np.matrix(xa)
        xs[0,0] = xs[0,0] / np.pi
        xs[0,1] = 2.0/(1.0 + math.exp(-0.25*xs[0,1])) - 1.0
        # xs[0,2] = xs[0,2] / np.pi
        # xs[0,3] = xs[0,3] / 20.0
        return self.model.predict({'state': xs, 'action': xa}, batch_size=len(xs))['value_output']

    def variables(self):
        return self.model.get_weights()

    def save(self, filepath):
        self.model.save_weights(filepath, overwrite=False)

    def restore(self, filepath):
        self.model.load_weights(filepath)

    def copy(self, scope=None):
        return ValueMLP(self.state_sizes, self.action_sizes, self.hiddens, self.nonlinearities, scope=scope, weights=self.model.get_weights(), regularizer=self.use_regularization)