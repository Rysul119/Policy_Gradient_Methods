#import libraries/frameworks
import gym
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer

#setting misc. properties
tf.keras.backend.set_floatx('float64')

#environment
env = gym.make('CartPole-v1')

#setting seed
env.seed(15)
np.random.seed(24)

#declaring constants
gamma = 0.99
upd = 5
policy_lr = 0.0005
value_lr = 0.001
clip_coeff = 0.1
lamb = 0.95
epochs = 1
episodes = 2000
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

#declaring models
#policy model
policy_model = tf.keras.Sequential()
policy_model.add(tf.keras.layers.Dense(64, input_shape = (state_dim, ), activation = 'relu'))
#policy_model.add(tf.keras.layers.Dense(64, activation = 'relu'))
policy_model.add(tf.keras.layers.Dense(act_dim, activation = 'softmax'))
policy_model_optimizer = tf.keras.optimizers.Adam(learning_rate = policy_lr)

#value model
value_model = tf.keras.Sequential()
value_model.add(tf.keras.layers.Dense(64, input_shape = (state_dim, ), activation = 'relu'))
#value_model.add(tf.keras.layers.Dense(64, activation = 'relu'))
value_model.add(tf.keras.layers.Dense(1, activation = 'linear'))
value_model_optimizer = tf.keras.optimizers.Adam(learning_rate = value_lr)

ep_score = []

for episode in range(episodes):
    states_batch, actions_batch, rewards_batch, act_probs_batch = [], [], [], []
    ep_reward = 0
    done = False
    state = env.reset()
    state = state.reshape([1, state_dim])
    while not done:
        #print(state)
        act_prob = policy_model.predict(state)
        #print(act_prob)
        action = np.random.choice(act_dim, p = act_prob[0])
        next_state, r, done, _ = env.step(action)

        next_state = np.reshape(next_state, [1, state_dim])

        states_batch.append(state[0])
        actions_batch.append(action)
        rewards_batch.append(r)
        act_probs_batch.append(act_prob[0])

        ep_reward += r
        state = next_state

        if len(states_batch) >= upd or done: #implement upd later
            '''batch = states_batch[0]
            for elem in states_batch[1:]:
                batch = np.append(batch, elem, axis = 0)

            print(batch)'''
            states_arr = np.array(states_batch)
            actions_arr = np.array(actions_batch)
            rewards_arr = np.array(rewards_batch)
            act_probs_arr = np.array(act_probs_batch)

            #GAE
            v = value_model.predict(states_arr)
            if done:
                next_v = 0
            else:
                #print(next_state)
                next_v = value_model.predict(next_state)

            td_targets = np.zeros_like(rewards_arr)
            gaes = np.zeros_like(rewards_arr)
            gae_cum = 0

            for i in reversed(range(0, len(rewards_arr))):
                delta = rewards_arr[i] + gamma * next_v - v[i]
                gae_cum = lamb * gamma * gae_cum + delta
                gaes[i] = gae_cum  #for policy loss calculation
                next_v = v[i]
                td_targets[i] = gaes[i] + v[i] #for value loss calculation

            td_targets = td_targets.reshape(len(rewards_batch), 1)
            #add multiple epochs for the update >>
            for epoch in range(epochs):
                #policy model update: policy loss and gradient calculation
                actions_onehot = tf.one_hot(actions_arr, act_dim, dtype= "float64")
                actions_onehot = tf.reshape(actions_onehot, [-1, act_dim])
                #actions_onehot = tf.cast(actions_onehot, tf.float64)
                #^^ check with prints
                with tf.GradientTape() as policy_tape:
                    act_probs_new = policy_model(states_arr, training =True)
                    gaes = tf.stop_gradient(gaes)
                    old_logp = tf.math.log(tf.reduce_sum(act_probs_arr * actions_onehot))
                    #print(actions_onehot)
                    old_logp = tf.stop_gradient(old_logp)
                    new_logp = tf.math.log(tf.reduce_sum(act_probs_new * actions_onehot))
                    prob_ratio = tf.math.exp(new_logp - old_logp)

                    clipped_ratio = tf.clip_by_value(prob_ratio, 1 - clip_coeff, 1 + clip_coeff)
                    policy_loss = -tf.minimum(prob_ratio * gaes, clipped_ratio * gaes)
                    policy_loss = tf.reduce_mean(policy_loss)

                policy_grads = policy_tape.gradient(policy_loss, policy_model.trainable_variables)
                #print(policy_grads)
                policy_model_optimizer.apply_gradients(zip(policy_grads, policy_model.trainable_variables))

                #Value model update: value loss and gradient calculation
                with tf.GradientTape() as value_tape:
                    v_pred = value_model(states_arr, training = True)
                    #print(v_pred.shape, td_targets.shape)
                    td_targets = tf.stop_gradient(td_targets)
                    assert v_pred.shape == td_targets.shape
                    #value_loss = tf.keras.losses.MeanSquaredError(td_targets, v_pred)
                    value_loss = tf.reduce_mean(tf.square(td_targets - v_pred)) * 0.5
                value_grads = value_tape.gradient(value_loss, value_model.trainable_variables)
                #print(value_grads)
                value_model_optimizer.apply_gradients(zip(value_grads, value_model.trainable_variables))

            states_batch, actions_batch, rewards_batch, act_probs_batch = [], [], [], []

    print("Episode: {} Score: {}".format(episode+1, ep_reward))
    ep_score.append(ep_reward)

dict = {'r': ep_score}
df = pd.DataFrame(dict)
x = np.arange(episodes)
plt.scatter(x + 1, df['r'], s=5)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("PPO on CartPole-v1")
plt.show()
plt.savefig('PPO_epi1k.png')