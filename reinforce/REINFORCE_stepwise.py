import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd



#episodes = 5000
tot_steps = 1000000
gamma = 0.98
runs = 20
for run in range(runs):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, input_dim=4, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    model.build()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    env = gym.make('CartPole-v1')
    env.seed(11+run)
    np.random.seed(6+run)

    scores = []
    steps =[]
    #step-wise gathering the trajectories and updating episode-wise
    #for e in range(episodes):
    s = env.reset()
    hist = []
    ep_score = 0
    ep_step = 0
    done = False
    episode = 0
    for step in range(tot_steps):
        s = s.reshape([1, 4])
        act_prob = model(s)
        act = np.random.choice(range(env.action_space.n), p=act_prob.numpy()[0])
        next_s, r, done, _ = env.step(act)
        hist.append((s, act, r))
        ep_score += r
        ep_step += 1
        s = next_s
        if done:
            #logging episode-wise rewards and steps
            scores.append(ep_score)
            steps.append(ep_step)
            R = 0
            for s, act, r in hist[::-1]:
                s = s.reshape([1, 4])
                R = r + gamma * R
                with tf.device('/GPU:1'):
                    with tf.GradientTape() as tape:
                        output = model(s)
                        #output = tf.reshape(output, [-1])
                        #actProbs = output[act]
                        actProbs = tf.gather(tf.reshape(output, [-1]), act)
                        loss = -tf.reduce_mean(tf.math.log(actProbs) * R)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
            #break
            episode+=1
            s = env.reset()
            hist =[]
            ep_score = 0
            ep_step = 0
            done = False
            if (episode+1) % 100 == 0:
                print("Episode {} avg rewards: {}".format(episode+1, np.mean(scores[-100:])))

    dict = {'r': scores, 'l': steps}
    df = pd.DataFrame(dict)
    #saving the logged rewards and steps in the log_dir directory
    log_dir = 'logs/CartPole-v1/lr/lr=0.01-'+str(run+1)
    os.makedirs(log_dir, exist_ok =True)
    file_name = 'monitor.csv'
    df.to_csv(log_dir+'/'+file_name)

x = np.cumsum(df['l'])
plt.scatter(x, df['r'])
plt.show()
