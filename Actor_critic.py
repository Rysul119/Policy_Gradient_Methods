import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

#check out the codes in stable basline using tf 1.x for a2c or any other compact 1.x codes for a2c

actor_model = tf.keras.Sequential()
actor_model.add(tf.keras.layers.Dense(64, input_shape = [None, 4], activation = 'relu'))
actor_model.add(tf.keras.layers.Dense(2, activation = 'softmax'))
actor_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)

critic_model = tf.keras.Sequential()
critic_model.add(tf.keras.layers.Dense(64, input_shape = [None, 4], activation = 'relu'))
critic_model.add(tf.keras.layers.Dense(1)) #starting with no activation
critic_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)

env = gym.make('CartPole-v1')
episodes = 10000
steps = 1000
gamma = 0.98
env.seed(10)
np.random.seed(5)

# trajectory collection to a certain size/horizon, calculate discounted rewards according to that and
# the algorithm of choice, also check with the algorithm pseudocode, log the state, actions, next state, rewards, dones
train_size = 64
scores =[]

for episode in range(episodes):
    s = env.reset()
    ep_score = 0
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []
    #done = False
    for step in range(steps):
        s = s.reshape([1, 4])
        actor_out = actor_model(s)
        #critic_out = critic_model(s)
        act = np.random.choice(range(env.action_space.n), p = actor_out.numpy()[0])
        next_s, r, done, _ = env.step(act)
        ep_score+=r
        states.append(s)
        next_states.append(next_s)
        actions.append(act)
        rewards.append(r)
        dones.append(done)
        s = next_s

        if len(states)==train_size:
            last_state = states[-1]
            if dones[-1] == True :
                reward_sum = 0
            else:
                reward_sum = critic_model(tf.convert_to_tensor(last_state[None, :], dtype=tf.float32))
            discounted_rewards = []
            for reward in rewards[::-1]:
                reward_sum = reward + gamma * reward_sum
                discounted_rewards.append(reward_sum)

            discounted_rewards.reverse()
            #^^got discounted rewards. have to think out incorporating in under the gradienttape.
            discounted_rewards_c = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32)

            #advantage calculation
            critic_out = critic_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            advntg = discounted_rewards - critic_out
            #updating critic parameters
            critic_variables = critic_model.trainable_variables
            with tf.GradientTape() as critic_tape:
                critic_tape.watch(critic_variables)
                critic_out = critic_model(tf.convert_to_tensor(np.vstack(states), dtype = tf.float32))
                critic_loss = tf.reduce_mean(tf.square(critic_out - discounted_rewards_c) * 0.5)
            critic_grads = critic_tape.gradient(critic_loss, critic_variables)
            critic_optimizer.apply_gradients(zip(critic_grads, critic_variables))

            #updating actor parameters
            actor_variables = actor_model.trainable_variables
            with tf.GradientTape() as actor_tape:
                actor_tape.watch(actor_variables)
                actor_out = actor_model(tf.convert_to_tensor(np.vstack(states), dtype = tf.float32))
                #assigning the actor_out probabilities according to the actions taken for loss calculation
                #have to clarify the matrix size below.

                #print(actor_out.numpy().shape)
                a = actor_out
                #print(tf.reshape(a, [-1]).numpy().shape)
                actProbs = tf.gather(tf.reshape(actor_out, [-1]), act)
                actor_loss = -tf.reduce_mean(tf.math.log(actProbs) * advntg)
                #add entropy loss^^
            actor_grads = actor_tape.gradient(actor_loss, actor_variables)
            actor_optimizer.apply_gradients(zip(actor_grads, actor_variables))
        if done:
            scores.append(ep_score)
            #print out episode last 50 episodes average reward. And if possible use tensorboard.
            break

    if (episode+1)%50 == 0:
        print("Episode: {} Average Reward: {}".format(episode+1, np.mean(scores[-50:])))


