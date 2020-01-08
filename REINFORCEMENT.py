import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_dim = 4, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation = "softmax"))
model.build()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)

env = gym.make('CartPole-v1')
episodes = 10000

score = 0
steps = 1000
gamma = 0.98

for e in range(episodes+1):
    s = env.reset()
    ep_memory = []
    history = []
    #ep_score = 0
    done = False
    for step in range(steps+1):
        s = s.reshape([1, 4])
        act_prob = model(s)

        act = np.random.choice(range(env.action_space.n), p=act_prob.numpy()[0])
        next_s, r, done, _ = env.step(act)
        history.append((s, act, r))
        score += r
        s = next_s
        if done:
            R = 0
            for s, act, r in history[::-1]:
                s = s.reshape([1, 4])
                R = r + gamma * R
                with tf.GradientTape() as tape:
                    output = model(s)
                    output = tf.reshape(output, [-1])
                    actProbs = output[act]
                    #actProbs = tf.gather(tf.reshape(output, [-1]), act)
                    loss = -tf.reduce_mean(tf.math.log(actProbs) * R)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            break
    if e % 50 == 0:
        print("Episode {} avg steps: {}".format(e, score/50))
        score = 0

dict = {'Score': ep_memory}
df = pd.DataFrame(dict)
df.to_csv('Ep_score.csv')

x = range(1, episodes+1)
plt.scatter(x, df['Score'])
plt.show()
