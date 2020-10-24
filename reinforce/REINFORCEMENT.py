import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, input_dim = 4, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation = "softmax"))
model.build()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

env = gym.make('CartPole-v1')
episodes = 10000
steps = 1000
gamma = 0.98
env.seed(10)
np.random.seed(5)

scores = []

for e in range(episodes):
    s = env.reset()
    hist = []
    ep_score = 0
    done = False
    for step in range(steps):
        s = s.reshape([1, 4])
        act_prob = model(s)
        act = np.random.choice(range(env.action_space.n), p=act_prob.numpy()[0])
        next_s, r, done, _ = env.step(act)
        hist.append((s, act, r))
        ep_score += r
        s = next_s
        if done:
            scores.append(ep_score)
            R = 0
            for s, act, r in hist[::-1]:
                s = s.reshape([1, 4])
                R = r + gamma * R
                with tf.GradientTape() as tape:
                    output = model(s)
                    #output = tf.reshape(output, [-1])
                    #actProbs = output[act]
                    actProbs = tf.gather(tf.reshape(output, [-1]), act)
                    loss = -tf.reduce_mean(tf.math.log(actProbs) * R)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            break
    if (e+1) % 100 == 0:
        print("Episode {} avg steps: {}".format(e+1, np.mean(scores[-100:])))

dict = {'Score': scores}
df = pd.DataFrame(dict)
df.to_csv('Reinforce.csv')

x = range(1, episodes+1)
plt.scatter(x, df['Score'])
plt.show()
