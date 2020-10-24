# a Q-actor critic algorithm

import gym
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

env = gym.make('Acrobot-v1')
np.random.seed(0)
tf.random.set_seed(0)
env.seed(0)

gamma = 0.99
total_steps = 1000000

# actor model
actor_model = tf.keras.Sequential()
actor_model.add(tf.keras.layers.Dense(100, input_shape = (env.observation_space.shape[0], ), activation='relu'))
actor_model.add(tf.keras.layers.Dense(env.action_space.n, activation='softmax'))
actor_model.build()
actor_optimizer = tf.keras.optimizers.Adam(lr=0.0005)

# critic model
critic_model = tf.keras.Sequential()
critic_model.add(tf.keras.layers.Dense(100, input_shape = (env.observation_space.shape[0], ), activation='relu'))
critic_model.add(tf.keras.layers.Dense(env.action_space.n, activation=tf.keras.activations.linear))
critic_model.build()
critic_optimizer = tf.keras.optimizers.Adam(lr=0.0005)


episode = 0
ep_score = 0
ep_step = 0
scores = []
steps = []

s = env.reset()
a_prob = actor_model(s[np.newaxis])
# print(a_prob)
a = np.random.choice(range(env.action_space.n), p=a_prob.numpy()[0])

for step in range(total_steps):
    next_s, r, done, _ = env.step(a)
    ep_score += r
    ep_step += 1
    next_a_prob = actor_model(next_s[np.newaxis])
    next_a = np.random.choice(range(env.action_space.n), p=next_a_prob.numpy()[0])
    u = critic_model(s[np.newaxis])
    q = u[0, a]

    # getting the actor model gradients and updating the weights
    with tf.GradientTape() as actor_tape:
        pi = actor_model(s[np.newaxis])[0, a]
        logpi = tf.math.log(pi)
        actor_loss = - logpi * q
    actor_grads = actor_tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor_model.trainable_variables))

    # gradientTape critic update:
    with tf.GradientTape() as critic_tape:
        q_old = critic_model(s[np.newaxis])
        q_next = critic_model(next_s[np.newaxis])[0, next_a]
        mask = np.zeros(tf.shape(q_old))
        mask[0, a] = 1
        # using masks to update q(a|s) values
        if done:
            r_tensor = tf.broadcast_to(r, tf.shape(q_old))
            target = tf.where(mask, r_tensor, q_old)
        else:
            r_tensor = tf.broadcast_to(r + gamma * q_next, tf.shape(q_old))
            target = tf.where(mask, r_tensor, q_old)
        critic_loss = tf.reduce_mean(tf.square(target - q_old)) * 0.5
    critic_grads = critic_tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))


    if done:
        episode += 1
        scores.append(ep_score)
        steps.append(ep_step)
        ep_score = 0
        ep_step = 0

        s = env.reset()
        if (episode % 100) == 0:
            print('Episode: {} Avg score: {}'.format(episode, np.mean(scores[-100:])))

    s, a = next_s, next_a

# saving the scores and corresposning steps
dict = {'r': scores, 'l': steps}
df = pd.DataFrame(dict)

# using cumulative sums to plot scores against steps taken
x = np.cumsum(df['l'])
plt.scatter(x, df['r'], s=5)
plt.show()