import numpy as np
import random
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from tf_rl.models import KERASMLP

from collections import deque

class KerasDDPG(object):
    def __init__(self, observation_size,
                       action_size,
                       actor,
                       critic,
                       exploration_sigma=0.001,
                       exploration_period=10000,
                       store_every_nth=5,
                       train_every_nth=5,
                       minibatch_size=64,
                       discount_rate=0.99,
                       max_experience=100000,
                       target_actor_update_rate=0.05,
                       target_critic_update_rate=0.05):
        """Initialized the Deepq object.

        Based on:
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        Parameters
        -------
        observation_size : int
            length of the vector passed as observation
        action_size : int
            length of the vector representing an action
        observation_to_actions: dali model
            model that implements activate function
            that can take in observation vector or a batch
            and returns scores (of unbounded values) for each
            action for each observation.
            input shape:  [batch_size, observation_size]
            output shape: [batch_size, action_size]
        exploration_sigma: float (0 to 1)
        exploration_period: int
            probability of choosing a random
            action (epsilon form paper) annealed linearly
            from 1 to exploration_sigma over
            exploration_period
        store_every_nth: int
            to further decorrelate samples do not all
            transitions, but rather every nth transition.
            For example if store_every_nth is 5, then
            only 20% of all the transitions is stored.
        train_every_nth: int
            normally training_step is invoked every
            time action is executed. Depending on the
            setup that might be too often. When this
            variable is set set to n, then only every
            n-th time training_step is called will
            the training procedure actually be executed.
        minibatch_size: int
            number of state,action,reward,newstate
            tuples considered during experience reply
        dicount_rate: float (0 to 1)
            how much we care about future rewards.
        max_experience: int
            maximum size of the reply buffer
        target_actor_update_rate: float
            how much to update target critci after each
            iteration. Let's call target_critic_update_rate
            alpha, target network T, and network N. Every
            time N gets updated we execute:
                T = (1-alpha)*T + alpha*N
        target_critic_update_rate: float
            analogous to target_actor_update_rate, but for
            target_critic
        """
        # memorize arguments
        self.observation_size = observation_size
        self.action_size = action_size

        self.actor = actor
        self.critic = critic
        self.target_actor = self.actor.copy()
        self.target_critic = self.critic.copy()

        self.exploration_sigma = exploration_sigma
        self.exploration_period = exploration_period
        self.store_every_nth = store_every_nth
        self.train_every_nth = train_every_nth
        self.minibatch_size = minibatch_size
        self.discount_rate = discount_rate
        self.learning_rate = 0.001
        self.max_experience = max_experience

        self.target_actor_update_rate = target_actor_update_rate
        self.target_critic_update_rate = target_critic_update_rate

        # deepq state
        self.actions_executed_so_far = 0
        self.experience = deque()

        self.iteration = 0

        self.number_of_times_store_called = 0
        self.number_of_times_train_called = 0

        # Orstein-Uhlenbeck Process (temporally correlated noise for exploration)
        self.noise_mean = 0.0
        self.noise_variance = 0.0
        self.ou_theta = 0.15
        self.ou_sigma = 0.2

        g_p = [K.gradients(K.sum(self.actor.model.get_output(train=False)), z)[0] for z in self.actor.model.trainable_weights]
        self.policy_gradient = K.function(inputs=[self.actor.model.get_input(train=False)], outputs=g_p)

        g_q = [K.gradients(K.sum(self.critic.model.get_output(train=False)), self.critic.model.get_input(train=False))[0]]
        self.critic_gradient = K.function(inputs=[self.critic.model.get_input(train=False)], outputs=g_q)

        self.tensor_board_cb = callbacks.TensorBoard()

    @staticmethod
    def linear_annealing(n, total, p_initial, p_final):
        """Linear annealing between p_initial and p_final
        over total steps - computes value at step n"""
        if n >= total:
            return p_final
        else:
            return p_initial - (n * (p_initial - p_final)) / (total)

    def plot_critic_value_function(self):
        dx, dy = 0.05, 0.05
        y, x = np.mgrid[slice(-3.15, 3.15+dy, dy), slice(-3.15, 3.15+dx, dx)]

        states = np.empty((len(y)*len(y), self.observation_size))
        actions = np.zeros((len(y)*len(y), self.action_size))

        for i in range(len(x)):
            for j in range(len(y)):
                states[i*len(y)+j,0] = x[i,j]
                states[i*len(y)+j,1] = 0
                states[i*len(y)+j,2] = y[i,j]
                states[i*len(y)+j,3] = 0

        q_0v_angle = self.critic(np.concatenate((states, actions), axis=1))
        q_0v_angle_t = self.target_critic(np.concatenate((states, actions), axis=1))

        new_q = np.zeros_like(x)
        new_q_t = np.zeros_like(x)
        for i in range(len(q_0v_angle)):
            new_q[i/len(y), i%len(y)] = q_0v_angle[i]
            new_q_t[i/len(y), i%len(y)] = q_0v_angle_t[i]

        new_q = new_q[:-1, :-1]
        new_q_t = new_q_t[:-1, :-1]

        levels = MaxNLocator(nbins=15).tick_values(new_q.min(), new_q.max())
        levels_t = MaxNLocator(nbins=15).tick_values(new_q_t.min(), new_q_t.max())
        cmap = plt.get_cmap('PiYG')
        fig, (ax0, ax1) = plt.subplots(nrows=2)
        cf = ax0.contourf(x[:-1, :-1] + dx/2., y[:-1, :-1]+dy/2., new_q, levels=levels, cmap=cmap)
        fig.colorbar(cf, ax=ax0)
        ax0.set_title('Critic Q val (joint vels = 0, action = 0')

        cf_t = ax1.contourf(x[:-1, :-1] + dx/2., y[:-1, :-1]+dy/2., new_q_t, levels=levels_t, cmap=cmap)
        fig.colorbar(cf_t, ax=ax1)
        ax1.set_title('Target Critic Q val (joint vels = 0, action = 0')

        fig.tight_layout()
        plt.show()

    def plot_actor_policy(self):
        dx, dy = 0.05, 0.05
        y, x = np.mgrid[slice(-3.15, 3.15+dy, dy), slice(-3.15, 3.15+dx, dx)]

        states = np.empty((len(y)*len(y), self.observation_size))

        for i in range(len(x)):
            for j in range(len(y)):
                states[i*len(y)+j,0] = x[i,j]
                states[i*len(y)+j,1] = 0
                states[i*len(y)+j,2] = y[i,j]
                states[i*len(y)+j,3] = 0

        actions = self.actor(states)
        actions_t = self.target_actor(states)

        new_a = np.zeros_like(x)
        new_a_t = np.zeros_like(x)
        for i in range(len(actions)):
            new_a[i/len(y), i%len(y)] = actions[i]
            new_a_t[i/len(y), i%len(y)] = actions_t[i]

        new_a = new_a[:-1, :-1]
        new_a_t = new_a_t[:-1, :-1]

        levels = MaxNLocator(nbins=15).tick_values(new_a.min(), new_a.max())
        levels_t = MaxNLocator(nbins=15).tick_values(new_a_t.min(), new_a_t.max())
        cmap = plt.get_cmap('PiYG')
        fig, (ax0, ax1) = plt.subplots(nrows=2)
        cf = ax0.contourf(x[:-1, :-1] + dx/2., y[:-1, :-1]+dy/2., new_a, levels=levels, cmap=cmap)
        fig.colorbar(cf, ax=ax0)
        ax0.set_title('Actor Policy (joint vels = 0)')

        cf_t = ax1.contourf(x[:-1, :-1] + dx/2., y[:-1, :-1]+dy/2., new_a_t, levels=levels_t, cmap=cmap)
        fig.colorbar(cf_t, ax=ax1)
        ax1.set_title('Target Actor Policy (joint vels = 0)')

        fig.tight_layout()
        plt.show()

    def update_target_networks(self):
        # Update target critic network
        target_weights = np.matrix(self.target_critic.model.get_trainable_weight_values())
        source_weights = np.matrix(self.critic.model.get_trainable_weight_values())
        new_weights = self.target_critic_update_rate*source_weights + (1-self.target_critic_update_rate)*target_weights
        self.target_critic.model.set_trainable_weight_values(new_weights.tolist()[0])

        # Updated target actor network
        target_weights = np.matrix(self.target_actor.model.get_trainable_weight_values())
        source_weights = np.matrix(self.actor.model.get_trainable_weight_values())
        new_weights = self.target_actor_update_rate*source_weights + (1-self.target_actor_update_rate)*target_weights
        self.target_actor.model.set_trainable_weight_values(new_weights.tolist()[0])

    def action(self, observation, dt):
        # assert len(observation.shape) == 1, \
        #        "Action is performed based on single observation."
        action = self.actor(observation)
        self.actions_executed_so_far += 1
        if self.exploration_period > self.actions_executed_so_far:
            # Solution for Ornstein-Uhlenbeck Process found here http://planetmath.org/ornsteinuhlenbeckprocess
            self.noise_mean = 0  # action * np.exp(-self.ou_theta*dt)
            self.noise_variance = self.ou_sigma*self.ou_sigma/self.ou_theta * (1 - np.exp(-2*self.ou_theta*dt))
            # noise_sigma = KerasDDPG.linear_annealing(self.actions_executed_so_far, self.exploration_period, 1.0, self.noise_variance)
            action += np.random.normal(self.noise_mean, self.noise_variance, size=action.shape)
            action = np.clip(action, -1., 1.)
        return action

    def target_action(self, observation):
        action = self.target_actor(observation)
        return action

    def store(self, observation, action, reward, newobservation):
        self.experience.append((observation, action, reward, newobservation))
        if len(self.experience) > self.max_experience:
            self.experience.popleft()

    def training_step(self):
        if len(self.experience) < 100*self.minibatch_size:
            return

        # sample experience (need to be a two liner, because deque...)
        samples = random.sample(range(len(self.experience)), self.minibatch_size)
        samples = [self.experience[i] for i in samples]

        # batch states
        states = np.empty((len(samples), self.observation_size))
        newstates = np.empty((len(samples), self.observation_size))
        actions = np.zeros((len(samples), self.action_size))
        target_actions = np.zeros((len(samples), self.action_size))
        rewards = np.empty((len(samples),1))

        for i, (state, action, reward, newstate) in enumerate(samples):
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            newstates[i] = newstate

        for i, state in enumerate(newstates):
            target_actions[i] = self.target_action(state)

        target_y = rewards + self.discount_rate * self.target_critic(np.concatenate((newstates, target_actions), axis=1))
        assert isinstance(self.critic.model, Sequential)
        # Update critic
        # self.critic.model.fit(np.concatenate((states, actions), axis=1), target_y, batch_size=len(samples), nb_epoch=1, verbose=0, callbacks=[self.tensor_board_cb])
        self.critic.model.fit(np.concatenate((states, actions), axis=1), target_y, batch_size=len(samples), nb_epoch=1, verbose=0)

        # Update actor policy
        policy_actions = np.zeros((len(samples), self.action_size))
        for i, state in enumerate(states):
            policy_actions[i] = self.actor(state)

        xs = np.matrix(states[0])
        grad_accum = np.zeros_like(self.policy_gradient([xs]))
        grad_accum = grad_accum.reshape((1,12))
        for i, (state, action) in enumerate(zip(states, policy_actions)):
            critic_xs = np.matrix(np.concatenate((state, action), axis=1))
            critic_grad = self.critic_gradient([critic_xs])
            actor_xs = np.matrix(state)
            policy_grad = self.policy_gradient([actor_xs])
            grad_accum = np.add(grad_accum, (critic_grad[0][0][4]*np.matrix(policy_grad)))

        policy_weight_updates = (grad_accum/float(len(samples)))
        current_weights = np.matrix(self.actor.model.get_trainable_weight_values())
        new_weights = current_weights + (self.learning_rate*policy_weight_updates)
        self.actor.model.set_trainable_weight_values(new_weights.tolist()[0])

        self.update_target_networks()








