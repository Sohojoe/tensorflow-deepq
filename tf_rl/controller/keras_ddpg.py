import numpy as np
import random
import math
import time
from keras import optimizers
from keras import backend as K
from keras import callbacks

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from tf_rl.controller.controller import BaseController

from collections import deque

LOG_FILE_DIR = '/home/mderry/local_data/rl_logs/pendulum_'
FILE_EXT = '.png'


class KerasDDPG(BaseController):
    def __init__(self, observation_size,
                       action_size,
                       actor,
                       critic,
                       exploration_sigma=0.001,
                       exploration_period=10000,
                       store_every_nth=4,
                       train_every_nth=4,
                       minibatch_size=128,
                       discount_rate=0.99,
                       max_experience=1000000,
                       target_actor_update_rate=0.0001,
                       target_critic_update_rate=0.0001):
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
        BaseController.__init__(self)
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
        self.training_steps = 0

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
        self.max_variance = 1.0
        self.min_variance = 0.01
        self.ou_theta = 0.25
        self.ou_sigma = 0.15

        self.minibatch = K.variable(minibatch_size)
        self.policy_learning_rate = K.variable(self.learning_rate)
        # self.policy_updates = self.get_policy_updates()

        self.target_actor_lr = K.variable(self.target_actor_update_rate)
        self.target_critic_lr = K.variable(self.target_critic_update_rate)

        self.policy_updater = self.get_policy_updater(self.minibatch_size)

        self.target_critic_updates = self.get_target_critic_updates()
        self.target_critic_updater = K.function(inputs=[], outputs=[], updates=self.target_critic_updates)

        self.target_actor_updates = self.get_target_actor_updates()
        self.target_actor_updater = K.function(inputs=[], outputs=[], updates=self.target_actor_updates)

        self.tensor_board_cb = callbacks.TensorBoard()
        self.bellman_error = []
        self.q_values = []

        self.num_training_iters = 1

    def policy_gradients(self):
        c_grads = K.gradients(self.critic.model.get_output(train=False)['value_output'], self.critic.model.get_input(train=False)['action'])[0]
        p_grads = [K.gradients(self.actor.model.get_output(train=False), z, grad_ys=c_grads)[0] for z in self.actor.model.trainable_weights]
        for i, pg in enumerate(p_grads):
            p_grads[i] = pg / self.minibatch
        return p_grads

    def get_policy_updates(self):
        p_grads = self.policy_gradients()
        policy_updates = []
        for p, g in zip(self.actor.model.trainable_weights, p_grads):
            # clipped_g = tf.clip_by_value(g, -1.0, 1.0)
            new_p = p + self.policy_learning_rate * g
            policy_updates.append((p, new_p))
        return policy_updates

    def get_policy_updater(self, minibatch_size):
        action_input_name = 'action'
        batch_size = minibatch_size
        shared_input_names = set(self.actor.model.inputs.keys()).intersection(set(self.critic.model.inputs.keys()))
        critic_layer_cache = self.critic.model.layer_cache
        actor_layer_cache = self.actor.model.layer_cache
        self.critic.model.layer_cache = {}
        self.actor.model.layer_cache = {}
        for name in shared_input_names:
            self.critic.model.inputs[name].previous = self.actor.model.inputs[name]
            self.critic.model.inputs[action_input_name].previous = self.actor.model.outputs['policy_output']
        output = self.critic.model.get_output(train=True)['value_output']
        if K._BACKEND == 'tensorflow':
            grads = K.gradients(output, self.actor.model.trainable_weights)
            # grads[0] = tf.Print(grads[0], [grads[0]], "grads = ", summarize=10)
            grads = [g / float(batch_size) for g in grads]
        elif K._BACKEND == 'theano':
            import theano.tensor as T
            grads = T.jacobian(output.flatten(), self.actor.model.trainable_weights)
            grads = [K.mean(g, axis=0) for g in grads]
        else:
            raise RuntimeError('unknown backend')
        for name in shared_input_names:
            del self.critic.model.inputs[name].previous
        del self.critic.model.inputs[action_input_name].previous
        self.critic.model.layer_cache = critic_layer_cache
        self.actor.model.layer_cache = actor_layer_cache

        # We now have the gradients (`grads`) of the combined model wrt to the actor's weights and
        # the output (`output`). Compute the necessary updates using a clone of the actor's optimizer.
        optimizer = self.actor.model.optimizer
        clipnorm = optimizer.clipnorm if hasattr(optimizer, 'clipnorm') else 0.
        clipvalue = optimizer.clipvalue if hasattr(optimizer, 'clipvalue') else 0.

        def get_gradients(loss, params):
            # We want to follow the gradient, but the optimizer goes in the opposite direction to
            # minimize loss. Hence the double inversion.
            assert len(grads) == len(params)
            modified_grads = [-g for g in grads]
            if clipnorm > 0.:
                norm = K.sqrt(sum([K.sum(K.square(g)) for g in modified_grads]))
                modified_grads = [optimizers.clip_norm(g, clipnorm, norm) for g in modified_grads]
            if clipvalue > 0.:
                modified_grads = [K.clip(g, -clipvalue, clipvalue) for g in modified_grads]
            return modified_grads

        optimizer.get_gradients = get_gradients
        updates = optimizer.get_updates(self.actor.model.trainable_weights, self.actor.model.constraints, None)
        updates += self.actor.model.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        inputs = self.actor.model.get_input(train=True)
        if isinstance(inputs, dict):
            inputs = [inputs[name] for name in self.actor.model.input_order]
        if not isinstance(inputs, list):
            inputs = [inputs]
        return K.function(inputs, [output], updates=updates)

    def get_target_critic_updates(self):
        critic_updates = []
        for tp, p in zip(self.target_critic.model.trainable_weights, self.critic.model.trainable_weights):
            new_p = self.target_critic_lr * p + (1 - self.target_critic_lr) * tp
            critic_updates.append((tp, new_p))
        return critic_updates

    def get_target_actor_updates(self):
        actor_updates = []
        for tp, p in zip(self.target_actor.model.trainable_weights, self.actor.model.trainable_weights):
            new_p = self.target_actor_lr * p + (1 - self.target_actor_lr) * tp
            actor_updates.append((tp, new_p))
        return actor_updates

    @staticmethod
    def linear_annealing(n, total, p_initial, p_final):
        """Linear annealing between p_initial and p_final
        over total steps - computes value at step n"""
        if n >= total:
            return p_final
        else:
            return p_initial - (n * (p_initial - p_final)) / (total)

    def plot_bellman_residual(self, save=False, filename=None):
        fig = plt.figure()
        if len(self.bellman_error) > 10000:
            step = len(self.bellman_error) / 100
            plt.plot(self.bellman_error[1::step])
            plt.xlabel('Training Step (in 1000s)')
        elif len(self.bellman_error) > 1000:
            step = len(self.bellman_error) / 1000
            plt.plot(self.bellman_error[1::step])
            plt.xlabel('Training Step (in 100s)')
        else:
            plt.plot(self.bellman_error)
            plt.xlabel('Training Step')
        plt.ylabel('Bellman Residual')

        if save:
            plt.savefig(filename, dpi=600)
        else:
            plt.show()
        plt.close()

    def plot_q_values(self, save=False, filename=None):
        fig = plt.figure()
        if len(self.q_values) > 10000:
            step = len(self.q_values) / 100
            plt.plot(self.q_values[1::step])
            plt.xlabel('Training Step (in 1000s)')
        elif len(self.q_values) > 1000:
            step = len(self.q_values) / 1000
            plt.plot(self.q_values[1::step])
            plt.xlabel('Training Step (in 100s)')
        else:
            plt.plot(self.q_values)
            plt.xlabel('Training Step')
        plt.ylabel('Q-Values')

        if save:
            plt.savefig(filename, dpi=600)
        else:
            plt.show()
        plt.close()

    def plot_critic_value_function(self, save=False, filename=None):
        fig, axes = plt.subplots(3, 7)
        fig.set_tight_layout(True)
        fig.set_size_inches(17, 6)
        cmap = plt.get_cmap('PiYG')

        dx, dy = 0.05, 0.2
        y, x = np.mgrid[slice(-20.0, 20.0+dy, dy), slice(-np.pi, np.pi+dx, dx)]

        for ax, action_multiplier in zip(axes.flat, np.linspace(-1.0, 1.0, num=21)):
            states = np.zeros((len(x[0])*len(x), self.observation_size))
            actions = np.ones((len(x[0])*len(x), self.action_size)) * action_multiplier

            for i in range(len(x)):
                for j in range(len(x[0])):
                    # print i*len(x)+j
                    states[i*len(x[0])+j,0] = x[i,j] / math.pi
                    #states[i*len(y)+j,1] = 0
                    states[i*len(x[0])+j,1] = 2.0 / (1.0 + math.exp(-0.25*y[i, j])) - 1.0  # y[i,j] / (2.0 * math.pi)
                    # states[i*len(y)+j,3] = 0

            q_0v_angle = self.critic(states, actions)

            new_q = np.zeros_like(x)
            for i in range(len(q_0v_angle)):
                new_q[i/len(x[0]), i%len(x[0])] = q_0v_angle[i]

            new_q = new_q[:-1, :-1]
            levels = MaxNLocator(nbins=15).tick_values(new_q.min(), new_q.max())

            cf = ax.contourf(x[:-1, :-1] + dx/2., y[:-1, :-1]+dy/2., new_q, levels=levels, cmap=cmap)
            fig.colorbar(cf, ax=ax)
            ax.set_title('action = %0.1f' % (action_multiplier))

        if save:
            plt.savefig(filename, dpi=600)
        else:
            plt.show()
        plt.close()

    def plot_actor_policy(self, save=False, filename=None):
        dx, dy = 0.05, 0.2
        y, x = np.mgrid[slice(-20.0, 20.0+dy, dy), slice(-np.pi, np.pi+dx, dx)]

        states = np.zeros((len(x[0])*len(x), self.observation_size))

        for i in range(len(x)):
            for j in range(len(x[0])):
                states[i*len(x[0])+j,0] = x[i,j] / math.pi
                # states[i*len(y)+j,1] = 0
                states[i*len(x[0])+j,1] = 2.0 / (1.0 + math.exp(-0.25*y[i, j])) - 1.0  # y[i,j] / (2.0 * math.pi)
                # states[i*len(y)+j,3] = 0

        actions = self.actor(states)
        actions_t = self.target_actor(states)

        new_a = np.zeros_like(x)
        new_a_t = np.zeros_like(x)
        for i in range(len(actions)):
            new_a[i/len(x[0]), i%len(x[0])] = actions[i]
            new_a_t[i/len(x[0]), i%len(x[0])] = actions_t[i]

        new_a = new_a[:-1, :-1]
        new_a_t = new_a_t[:-1, :-1]

        levels = MaxNLocator(nbins=15).tick_values(new_a.min(), new_a.max())
        levels_t = MaxNLocator(nbins=15).tick_values(new_a_t.min(), new_a_t.max())
        cmap = plt.get_cmap('PiYG')
        fig, (ax0, ax1) = plt.subplots(nrows=2)
        cf = ax0.contourf(x[:-1, :-1] + dx/2., y[:-1, :-1]+dy/2., new_a, levels=levels, cmap=cmap)
        ax0.set_xlabel('Joint Angle')
        ax0.set_ylabel('Joint Velocity')
        fig.colorbar(cf, ax=ax0)
        ax0.set_title('Actor Policy')

        cf_t = ax1.contourf(x[:-1, :-1] + dx/2., y[:-1, :-1]+dy/2., new_a_t, levels=levels_t, cmap=cmap)
        fig.colorbar(cf_t, ax=ax1)
        ax1.set_xlabel('Joint Angle')
        ax1.set_ylabel('Joint Velocity')
        ax1.set_title('Target Actor Policy')

        fig.tight_layout()
        if save:
            plt.savefig(filename, dpi=600)
        else:
            plt.show()
        plt.close()

    def action(self, observation, dt, ignore_exploration=False):
        # assert len(observation.shape) == 1, \
        #        "Action is performed based on single observation."
        action = self.actor(observation)
        self.actions_executed_so_far += 1
        if not ignore_exploration:
            # Solution for Ornstein-Uhlenbeck Process found here http://planetmath.org/ornsteinuhlenbeckprocess
            self.noise_mean = 0  # action * np.exp(-self.ou_theta*dt)
            # self.noise_variance = self.ou_sigma*self.ou_sigma/self.ou_theta * (1 - np.exp(-2*self.ou_theta*dt))
            noise_sigma = KerasDDPG.linear_annealing(self.actions_executed_so_far, self.exploration_period, self.max_variance, self.min_variance)
            action += np.random.normal(self.noise_mean, noise_sigma, size=action.shape)
            action = np.clip(action, -1., 1.)
        return action
        # return 0.0

    def target_action(self, observation):
        action = self.target_actor(observation)
        return action

    def store(self, observation, action, reward, newobservation):
        self.experience.append((observation, action, reward, newobservation))
        if len(self.experience) > self.max_experience:
            self.experience.popleft()

    def save_checkpoint(self, filepath):
        import pickle
        self.actor.save('%s' % (filepath + '_actor'))
        self.target_actor.save('%s' % (filepath + '_target_actor'))
        self.critic.save('%s' % (filepath + '_critic'))
        self.target_critic.save('%s' % (filepath + '_target_critic'))
        with open('%s' % (filepath + '_exp'), 'w') as exp_file:
            pickle.dump(self.experience, exp_file)

    def restore_checkpoint(self, filepath, restore_exp=False):
        import pickle
        self.actor.restore('%s' % (filepath + '_actor'))
        self.target_actor.restore('%s' % (filepath + '_target_actor'))
        self.critic.restore('%s' % (filepath + '_critic'))
        self.target_critic.restore('%s' % (filepath + '_target_critic'))
        if restore_exp:
            with open('%s' % (filepath + '_exp'), 'r') as exp_file:
                self.experience = pickle.load(exp_file)

    def training_step(self):
        start_time = time.time()
        if len(self.experience) < 20*self.minibatch_size:
            return

        self.training_steps += 1
        print 'Starting training step %d at %s' % (self.training_steps, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)))

        bellman_residual = []
        q_values = []

        for k in range(self.num_training_iters):
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
            target_y = rewards + self.discount_rate * self.target_critic(newstates, target_actions)
            # Update critic
            # self.critic.model.fit(np.concatenate((states, actions), axis=1), target_y, batch_size=len(samples), nb_epoch=1, verbose=0, callbacks=[self.tensor_board_cb])
            history = self.critic.model.fit({'state': states, 'action': actions, 'value_output': target_y}, batch_size=len(samples), nb_epoch=1, verbose=0)
            bellman_residual.append(history.history['loss'][0])

            # Update actor policy
            # policy_actions = np.zeros((len(samples), self.action_size))
            # for i, state in enumerate(states):
            #     policy_actions[i] = self.actor(state)
            # # critic_xs = np.matrix(np.concatenate((states, policy_actions), axis=1))
            # actor_xs = np.matrix(states)
            # self.policy_updater([np.matrix(states), np.matrix(policy_actions), actor_xs])
            states = states.tolist()
            # print type(states)
            # print states
            # self.policy_updater = self.get_policy_updater(self.minibatch_size)
            q_val = self.policy_updater([states])[0].flatten()
            q_values.append(np.mean(q_val))

            self.target_critic_updater([])
            self.target_actor_updater([])

        self.bellman_error.append(np.mean(bellman_residual))
        self.q_values.append(np.mean(q_values))

        if self.training_steps % 2000 == 1:
            self.plot_critic_value_function(save=True, filename='%s' % (LOG_FILE_DIR + 'critic_' + str(self.training_steps) + FILE_EXT))
            self.plot_actor_policy(save=True, filename='%s' % (LOG_FILE_DIR + 'policy_' + str(self.training_steps) + FILE_EXT))
            self.plot_bellman_residual(save=True, filename='%s' % (LOG_FILE_DIR + 'bellman_residual_' + str(self.training_steps) + FILE_EXT))
            self.plot_q_values(save=True, filename='%s' % (LOG_FILE_DIR + 'q_values_' + str(self.training_steps) + FILE_EXT))
            self.save_checkpoint(filepath='%s' % (LOG_FILE_DIR + 'checkpoint_' + str(self.training_steps)))
