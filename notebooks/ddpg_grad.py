from keras import backend as K

actor = None # the following code assumes that actor and critic are Graph networks
critic = None
action_input_name = 'input_action'
output_name = 'output'
batch_size = 64

# Temporarily connect to a large, combined model so that we can compute the gradient and monitor
# the performance of the actor as evaluated by the critic.
shared_input_names = set(actor.inputs.keys()).intersection(set(critic.inputs.keys()))
critic_layer_cache = critic.layer_cache
actor_layer_cache = actor.layer_cache
critic.layer_cache = {}
actor.layer_cache = {}
for name in shared_input_names:
    critic.inputs[name].previous = actor.inputs[name]
critic.inputs[action_input_name].previous = actor.outputs[output_name]
output = critic.get_output(train=True)[output_name]
if K._BACKEND == 'tensorflow':
    grads = K.gradients(output, actor.trainable_weights)
    grads = [g / float(batch_size) for g in grads]
elif K._BACKEND == 'theano':
    import theano.tensor as T
    grads = T.jacobian(output.flatten(), actor.trainable_weights)
    grads = [K.mean(g, axis=0) for g in grads]
else:
    raise RuntimeError('unknown backend')
for name in shared_input_names:
    del critic.inputs[name].previous
del critic.inputs[action_input_name].previous
critic.layer_cache = critic_layer_cache
actor.layer_cache = actor_layer_cache

# We now have the gradients (`grads`) of the combined model wrt to the actor's weights and
# the output (`output`). Compute the necessary updates using a clone of the actor's optimizer.
optimizer = actor.optimizer
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
updates = optimizer.get_updates(actor.trainable_weights, actor.constraints, None)
updates += actor.updates # include other updates of the actor, e.g. for BN

# Finally, combine it all into a callable function.
inputs = actor.get_input(train=True)
if isinstance(inputs, dict):
    inputs = [inputs[name] for name in actor.input_order]
assert isinstance(inputs, list)
fn = K.function(inputs, [output], updates=updates)

# At this point, fn can be used to train the actor.
# It will also perform a forward pass to compute the (estimated) Q value
# of the action wrt to the inputs. The forward pass is not necesssary
# per-se, but it proved to be a good indicator for convergence during training.

# Train the actor on a single mini-batch.
state0_batch = {} # get data from replay memory here
q_values = fn([state0_batch[name] for name in actor.input_order])[0].flatten()
