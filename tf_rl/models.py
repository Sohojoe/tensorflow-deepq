import math
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
import numpy as np


def custom_initialization(shape, scale=0.003, name=None):
    return K.variable(np.random.uniform(low=-scale, high=scale, size=shape), name=name)


class DDPGPolicyMLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities, scope=None, weights=None, regularizer=None, learning_rate=0.0001):
        self.input_sizes = input_sizes

        self.hiddens = hiddens
        self.nonlinearities = nonlinearities
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        self.scope = scope or "MLP"
        self.input_weights = weights
        self.W_regularizer = regularizer

        assert len(hiddens) == len(nonlinearities), \
            "Number of hiddens must be equal to number of nonlinearities"

        self.model = Graph()
        self.model.add_input('state', input_shape=(self.input_sizes,))
        self.model.add_node(Dense(hiddens[0], init='lecun_uniform', W_regularizer=self.W_regularizer), name='layer_0', input='state')
        self.model.add_node(Activation('relu'), name='activation_0', input='layer_0')
        last_layer = 0
        for l_idx, (h_from, h_to) in enumerate(zip(hiddens[:-2], hiddens[1:-1])):
            self.model.add_node(Dense(h_to, init='lecun_uniform', W_regularizer=self.W_regularizer), name='layer_%d' % (l_idx+1), input='activation_%d' % l_idx)
            self.model.add_node(Activation('relu'), name='activation_%d' % (l_idx+1), input='layer_%d' % (l_idx+1))
            last_layer = l_idx+1

        self.model.add_node(Dense(hiddens[-1], init='lecun_uniform', W_regularizer=self.W_regularizer), name='layer_out', input='layer_%d' % last_layer)
        self.model.add_node(Activation('tanh'), name='activation_out', input='layer_out')
        self.model.add_output(name='policy_output', input='activation_out')

        self.opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss={'policy_output': 'MSE'}, optimizer=self.opt)
        if weights is not None:
            self.model.set_weights(weights)

    def __call__(self, xs):
        if len(xs) > 1:
            xs = np.matrix(xs)
        xs[0, 0] = xs[0, 0] / np.pi
        xs[0, 1] = 2.0/(1.0 + math.exp(-0.25*xs[0, 1])) - 1.0
        xs[0, 2] = xs[0, 2] / np.pi
        xs[0, 3] = 2.0/(1.0 + math.exp(-0.25*xs[0, 3])) - 1.0
        return self.model.predict({'state': xs}, batch_size=len(xs))['policy_output']

    def variables(self):
        return self.model.get_weights()

    def save(self, filepath):
        self.model.save_weights(filepath, overwrite=False)

    def restore(self, filepath):
        self.model.load_weights(filepath)

    def copy(self, scope=None):
        return DDPGPolicyMLP(self.input_sizes, self.hiddens, self.nonlinearities, scope=scope, weights=self.model.get_weights(), regularizer=self.W_regularizer)


class DDPGValueMLP(object):
    def __init__(self, state_sizes, action_sizes, hiddens, nonlinearities, scope=None, weights=None, regularizer=None, learning_rate=0.001):
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
        self.W_regularizer = regularizer

        assert len(hiddens) == len(nonlinearities), \
            "Number of hiddens must be equal to number of nonlinearities"

        # Create Graph for full model
        self.model = Graph()
        self.model.add_input('state', input_shape=(self.state_sizes,))
        self.model.add_input('action', input_shape=(self.action_sizes,))

        # Create Sequential container for state processing
        self.state_proc = Sequential()
        self.state_proc.add(Dense(hiddens[0], input_dim=self.state_sizes, init="lecun_uniform", W_regularizer=self.W_regularizer))
        self.state_proc.add(Activation('relu'))

        # Add State Processer to Graph
        self.model.add_node(self.state_proc, name='state_proc', input='state')

        # Add Dense Layer to combine state processor with action input
        self.model.add_node(Dense(hiddens[1], init='lecun_uniform', W_regularizer=self.W_regularizer), name='joiner', inputs=['state_proc', 'action'])
        self.model.add_node(Activation('relu'), name='activation_0', input='joiner')

        if len(self.hiddens) > 3:
            # Add remaining hidden layers
            last_idx = 0
            for l_idx, (h_from, h_to) in enumerate(zip(hiddens[1:-2], hiddens[2:-1])):
                self.model.add_node(Dense(h_to, init='lecun_uniform', W_regularizer=self.W_regularizer), name='dense_%d' % (l_idx+1), input='activation_%d' % l_idx)
                self.model.add_node(Activation('relu'), name='activation_%d' % (l_idx+1), input='dense_%d' % (l_idx+1))
                last_idx = l_idx

            self.model.add_node(Dense(hiddens[-1], init=custom_initialization, W_regularizer=self.W_regularizer), name='dense_out', input='activation_%d' % (last_idx+1))
            self.model.add_node(Activation('relu'), name='activation_out', input='dense_out')
        else:
            self.model.add_node(Dense(hiddens[-1], init=custom_initialization, W_regularizer=self.W_regularizer), name='dense_out', input='activation_0')
            self.model.add_node(Activation('linear'), name='activation_out', input='dense_out')

        self.model.add_output(name='value_output', input='activation_out')

        self.opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss={'value_output': 'MSE'}, optimizer=self.opt)
        if weights is not None:
            self.model.set_weights(weights)

    def __call__(self, xs, xa):
        if len(xs) > 1:
            xs = np.matrix(xs)
        if len(xa) > 1:
            xa = np.matrix(xa)
        xs[0, 0] = xs[0, 0] / np.pi
        xs[0, 1] = 2.0/(1.0 + math.exp(-0.25*xs[0, 1])) - 1.0
        xs[0, 2] = xs[0, 2] / np.pi
        xs[0, 3] = 2.0/(1.0 + math.exp(-0.25*xs[0, 3])) - 1.0
        return self.model.predict({'state': xs, 'action': xa}, batch_size=len(xs))['value_output']

    def variables(self):
        return self.model.get_weights()

    def save(self, filepath):
        self.model.save_weights(filepath, overwrite=False)

    def restore(self, filepath):
        self.model.load_weights(filepath)

    def copy(self, scope=None):
        return DDPGValueMLP(self.state_sizes, self.action_sizes, self.hiddens, self.nonlinearities, scope=scope, weights=self.model.get_weights(), regularizer=self.W_regularizer)
