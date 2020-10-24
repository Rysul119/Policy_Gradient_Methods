#importing libraries
import tensorflow as tf
from tensorflow.keras.layers import  Input, Dense, Lambda

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pybulletgym
import pybullet_envs

tf.keras.backend.set_floatx('float64')

#constants

gamma = 0.99
upd = 1024
policy_lr = 0.0001
value_lr = 0.0005
env_name = 'HopperBulletEnv-v0'
env = gym.make(env_name)
env.seed(12)
np.random.seed(15)
max_episodes = 3000

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
std_bound = [1e-2, 1.0]

ep_score = []

#models
#policy model
state_input = Input((state_dim,))
dense_1 = Dense(64, activation='relu')(state_input)
dense_2 = Dense(64, activation='relu')(dense_1)
out_mu = Dense(action_dim, activation='tanh')(dense_2)
mu_output = Lambda(lambda x: x * action_bound)(out_mu)
std_output = Dense(action_dim, activation='softplus')(dense_2)
policy_model = tf.keras.models.Model(state_input, [mu_output, std_output])
policy_model_optimize = tf.keras.optimizers.Adam(policy_lr)

#value model
value_model = tf.keras.Sequential([
    Input((state_dim,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])
value_model_optimizer = tf.keras.optimizers.Adam(value_lr)

for episode in range(max_episodes):
    state_batch = []
    action_batch = []
    td_target_batch = []
    advantage_batch = []
    episode_reward = 0
    done = False

    state = env.reset()

    while not done:
        state = np.reshape(state, [1, state_dim])
        mu, std = policy_model.predict(state)
        action = np.random.normal(mu[0], std[0], size=action_dim)
        action = np.clip(action, -action_bound, action_bound)

        next_state, reward, done, _ = env.step(action)

        state = np.reshape(state, [1, state_dim])
        action = np.reshape(action, [1, action_dim])
        next_state = np.reshape(next_state, [1, state_dim])
        reward = np.reshape(reward, [1, 1])

        if done:
            td_target = reward
        else:
            v_value = value_model.predict((np.reshape(next_state, [1, state_dim])))
            td_target = np.reshape(reward + gamma * v_value[0], [1, 1])

        advantage = td_target - value_model.predict(state)

        state_batch.append(state)
        action_batch.append(action)
        td_target_batch.append(td_target)
        advantage_batch.append(advantage)

        if len(state_batch) >= upd or done:

            states_arr = state_batch[0]
            for elem in state_batch[1:]:
                states_arr = np.append(states_arr, elem, axis = 0)

            actions_arr = action_batch[0]
            for elem in action_batch[1:]:
                actions_arr = np.append(actions_arr, elem, axis=0)

            td_targets_arr = td_target_batch[0]
            for elem in td_target_batch[1:]:
                td_targets_arr = np.append(td_targets_arr, elem, axis=0)

            advantages_arr = advantage_batch[0]
            for elem in advantage_batch[1:]:
                advantages_arr = np.append(advantages_arr, elem, axis=0)

            #gradient and loss calculation
            #policy
            with tf.GradientTape() as policy_tape:
                mus, stds = policy_model(states_arr, training=True)
                stds = tf.clip_by_value(stds, std_bound[0], std_bound[1])
                vars = stds ** 2
                log_policy_pdf = -0.5 * (actions_arr - mus) ** 2 / \
                                 vars - 0.5 * tf.math.log(vars * 2 * np.pi)
                log_policy_pdf = tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

                policy_loss = log_policy_pdf * advantages_arr
                policy_loss = tf.reduce_sum(-policy_loss)
            policy_grads = policy_tape.gradient(policy_loss, policy_model.trainable_variables)
            policy_model_optimize.apply_gradients(zip(policy_grads, policy_model.trainable_variables))

            #value
            with tf.GradientTape() as value_tape:
                v_pred = value_model(states_arr, training=True)
                assert v_pred.shape == td_targets_arr.shape
                mse = tf.keras.losses.MeanSquaredError()
                value_loss = mse(v_pred, tf.stop_gradient(td_targets_arr))
            value_grads = value_tape.gradient(value_loss, value_model.trainable_variables)
            value_model_optimizer.apply_gradients(zip(value_grads, value_model.trainable_variables))

            state_batch = []
            action_batch = []
            td_target_batch = []
            advantage_batch = []

        episode_reward += reward[0][0]
        state = next_state[0]

    ep_score.append(episode_reward)
    print('EP{} EpisodeReward={}'.format(episode+1, episode_reward))

dict = {'r': ep_score}
df = pd.DataFrame(dict)
x = np.arange(max_episodes)
plt.scatter(x, df['r'], s = 5)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('A2C on '+env_name)
plt.show()