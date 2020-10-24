#importing frameworks/libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
import pandas as pd
import matplotlib.pyplot as plt
import gym
import numpy as np
from timeit import default_timer as timer
import os
import pybulletgym
import pybullet_envs

tf.keras.backend.set_floatx('float64')


#declaring constants
gamma = 0.99
upd = 1024
epochs = 15
clip_coeff = 0.1
lamb = 0.95
#max_episodes = 3000
tot_steps = 2000000


#declaring env
env_name = 'Walker2DBulletEnv-v0'
env = gym.make(env_name)
env.seed(15)
np.random.seed(24)
tf.random.set_seed(34)
node = 64

#getting the dimension of the environment
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_bound = env.action_space.high[0]
#act_bound_low = env.action_space.low[0]
#print(act_bound, act_bound_low)
std_bound = [1e-2, 1.0]

#declaring the model
#policy model (functional api) + optimizer
state_input = Input((state_dim, ))
'''dense_1 = Dense(state_dim * 10, activation='relu')(state_input)
dense_2 = Dense(int(np.sqrt(state_dim * 10 * act_dim * 10)), activation='relu')(dense_1)
dense_3 = Dense(act_dim * 10, activation='relu')(dense_2)'''
dense_1 = Dense(node, activation='relu')(state_input)
#dense_2 = Dense(2 * node, activation='relu')(dense_1)
#dense_3 = Dense(node, activation='relu')(dense_2)
int_mu = Dense(act_dim, activation='tanh')(dense_1)
mu = Lambda(lambda x: x * act_bound)(int_mu)
std = Dense(act_dim, activation='softplus')(dense_1)
policy_model = Model(state_input, [mu, std])

#policy_lr = 9e-4 / np.sqrt(int(np.sqrt(state_dim * 10 * act_dim * 10)))
policy_lr = 0.0001
print(policy_lr)
policy_model_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_lr)

#value model
'''value_model = tf.keras.Sequential([
    Input((state_dim, )),
    Dense(state_dim * 10, activation='relu'),
    Dense(int(np.sqrt(state_dim * 10 * 10)), activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])'''

value_model = tf.keras.Sequential([
    Input((state_dim, )),
    Dense(node, activation='relu'),
    #Dense(node, activation='relu'),
    #Dense(node, activation='relu'),
    Dense(1, activation='linear')
])

#value_lr = 1e-2 / np.sqrt(int(np.sqrt(state_dim * 10 * 5)))
value_lr = 0.0002
print(value_lr)
value_model_optimizer = tf.keras.optimizers.Adam(learning_rate=value_lr)

#sampling and training
ep_score = 0
ep_step = 0
episode = 0
scores = []
steps = []
#make it stepwise
state_batch = []
action_batch = []
reward_batch = []
old_policy_batch = []
done =  False
env.render()
state = env.reset()

print("Experiment is starting to learn {} using PPO with a policy learning rate {} and value learning rate {}.".format(env_name, policy_lr, value_lr))
start = timer()

for step in range(tot_steps):
    state = np.reshape(state, [1, state_dim])
    mu, std = policy_model.predict(state)
    action = np.random.normal(mu[0], std[0], size=act_dim)
    action = np.clip(action, -act_bound, act_bound)
    std = tf.clip_by_value(std, std_bound[0], std_bound[1])
    var = std ** 2
    log_policy_pdf = -0.5 * (action - mu) ** 2 / \
                     var - 0.5 * tf.math.log(var * 2 * np.pi)
    log_policy = tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    next_state, reward, done, _ = env.step(action)

    state = np.reshape(state, [1, state_dim])
    action = np.reshape(action, [1, act_dim])
    next_state = np.reshape(next_state, [1, state_dim])
    reward = np.reshape(reward, [1, 1])
    log_policy = np.reshape(log_policy, [1, 1])

    state_batch.append(state)
    action_batch.append(action)
    reward_batch.append(reward)
    old_policy_batch.append(log_policy)

    if len(state_batch) >= upd or done:
        # list to batch
        states_arr = state_batch[0]
        for elem in state_batch[1:]:
            states_arr = np.append(states_arr, elem, axis=0)
        # normalizing states
        #states_arr = (states_arr - states_arr.mean()) / states_arr.std()

        actions_arr = action_batch[0]
        for elem in action_batch[1:]:
            actions_arr = np.append(actions_arr, elem, axis=0)
        # normalizing actions
        #actions_arr = (actions_arr - actions_arr.mean()) / actions_arr.std()

        rewards_arr = reward_batch[0]
        for elem in reward_batch[1:]:
            rewards_arr = np.append(rewards_arr, elem, axis=0)
        # normalizing rewards
        #rewards_arr = (rewards_arr - rewards_arr.mean())/rewards_arr.std()

        old_policies_arr = old_policy_batch[0]
        for elem in old_policy_batch[1:]:
            old_policies_arr = np.append(old_policies_arr, elem, axis=0)

        v_values = value_model.predict(states_arr)
        next_v_value = value_model.predict(next_state)

        td_targets = np.zeros_like(rewards_arr)
        gae = np.zeros_like(rewards_arr)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for i in reversed(range(0, len(rewards_arr))):
            delta = rewards_arr[i] + gamma * forward_val - v_values[i]
            gae_cumulative = gamma * lamb * gae_cumulative + delta
            gae[i] = gae_cumulative
            forward_val = v_values[i]
            td_targets[i] = gae[i] + v_values[i]

        for epoch in range(epochs):
            # policy model train
            with tf.GradientTape() as policy_tape:
                mus, stds = policy_model(states_arr, training=True)
                stds = tf.clip_by_value(stds, std_bound[0], std_bound[1])
                vars = stds ** 2
                new_log_policy_pdf = -0.5 * (actions_arr - mus) ** 2 / \
                                     vars - 0.5 * tf.math.log(vars * 2 * np.pi)
                new_log_policy = tf.reduce_sum(new_log_policy_pdf, 1, keepdims=True)
                prob_ratio = tf.exp(new_log_policy - tf.stop_gradient(old_policies_arr))
                gae = tf.stop_gradient(gae)
                clipped_ratio = tf.clip_by_value(prob_ratio, 1.0 - clip_coeff, 1 + clip_coeff)
                surrogate = -tf.minimum(prob_ratio * gae, clipped_ratio * gae)
                policy_loss = tf.reduce_mean(surrogate)
            policy_grads = policy_tape.gradient(policy_loss, policy_model.trainable_variables)
            policy_model_optimizer.apply_gradients(zip(policy_grads, policy_model.trainable_variables))

            # value model train
            with tf.GradientTape() as value_tape:
                v_pred = value_model(states_arr, training=True)
                assert v_pred.shape == td_targets.shape
                mse = tf.keras.losses.MeanSquaredError()
                value_loss = mse(v_pred, tf.stop_gradient(td_targets))
            value_grads = value_tape.gradient(value_loss, value_model.trainable_variables)
            value_model_optimizer.apply_gradients(zip(value_grads, value_model.trainable_variables))

        state_batch = []
        action_batch = []
        reward_batch = []
        old_policy_batch = []

    ep_score += reward[0][0]
    ep_step += 1
    state = next_state[0]

    if done:
        scores.append(ep_score)
        steps.append(ep_step)
        episode += 1
        print("Episode: {} Step: {} Reward: {}".format(episode, step + 1, ep_score))
        ep_score = 0
        ep_step = 0
        state = env.reset()

end = timer()

print('Experiment is finished to learn {} using PPO with a policy learning rate {} and value learning rate {} after {} hours.'.format(env_name, policy_lr, value_lr, (end-start)/3600.0))
dict = {'r': scores, 'l': steps}
df = pd.DataFrame(dict)
x = np.cumsum(df['l'])
plt.scatter(x, df['r'], s = 5)
plt.xlabel('Steps')
plt.ylabel('Episodic Score')
plt.show()

