import gym
import tensorflow as tf
import numpy as np


env = gym.make('CartPole-v1')
episodes = 100000
steps = 1000
gamma = 0.98
env.seed(10)
np.random.seed(5)

scores = []

#actor model
actor_model = tf.keras.Sequential()
actor_model.add(tf.keras.layers.Dense(64, input_dim = 4, activation = 'relu'))
actor_model.add(tf.keras.layers.Dense(2, activation='softmax'))
actor_model.build()
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

#critic model
critic_model = tf.keras.Sequential()
critic_model.add(tf.keras.layers.Dense(64, input_dim=4, activation='relu'))
critic_model.add(tf.keras.layers.Dense(1))
critic_model.build()
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

#think about the implementation in steps with maxsteps/horizon

for episode in range(episodes):

    #reset the gradients for the actor and critic nn model
    grad_buffer_c = critic_model.trainable_variables
    for ix, grad in enumerate(grad_buffer_c):
        grad_buffer_c[ix] = grad * 0
    grad_buffer_a = actor_model.trainable_variables
    for ix, grad in enumerate(grad_buffer_a):
        grad_buffer_a[ix] = grad * 0

    #print(tf.shape(actor_model.trainable_variables[0]))
    #print(tf.shape(grad_buffer_a[0]))
    #print(grad_buffer_a)
    s = env.reset()
    hist = []
    ep_memoryc = []
    ep_memorya = []
    ep_score = 0
    done = False
    for step in range(steps):
        s = s.reshape([1, 4])
        act_prob = actor_model(s)
        act = np.random.choice(range(env.action_space.n), p = act_prob.numpy()[0])
        next_s, r, done, _ = env.step(act)
        hist.append((s, act, r))
        ep_score += r
        s = next_s

        if done:
            break

    #getting the last state stats
    last_s, last_done, _ = hist[-1]

    if last_done:
        scores.append(ep_score)
        R=0
    else:
        R=critic_model(last_s)

    for s, act, r in hist[::-2]:
        R = r + gamma * R
        #getting gradients for the value network
        with tf.GradientTape() as critic_tape:
            v = critic_model(s)
            critic_loss = tf.reduce_mean(tf.square(R - v)) * 0.5
        grads = critic_tape.gradient(critic_loss, critic_model.trainable_variables)
        #critic_optimizer.apply_gradients(zip(grads, critic_model.trainable_variables))
        #hold the grads for updates
        ep_memoryc.append(grads)

        #getting gradients for the policy network
        with tf.GradientTape() as actor_tape:
            pi = actor_model(s)
            pi_as = tf.gather(tf.reshape(pi, [-1]), act)
            logp = -tf.reduce_mean(tf.math.log(pi_as))
        grads = actor_tape.gradient(logp, actor_model.trainable_variables)
        adv = R-v
        #print(adv)
        #actor_optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))
        #hold the grads and R-v for updates
        ep_memorya.append([grads, adv])
    ep_memorya=np.array(ep_memorya)
    ep_memoryc=np.array(ep_memoryc)
    #accumulating gradients<<-keep checking on the error
    for grads in ep_memoryc:
        for ix, grad in enumerate(grads):
            grad_buffer_c[ix] += grad
    #problem with the size of grad_buffer_a
    #solved was having a product of wrong size when multiplied with adv which has the shape of 1,1. So indexing it with [0]
    #helped solving the problem
    for grads, adv in ep_memorya:
        for ix, grad in enumerate(grads):
            grad_buffer_a[ix] += grad * adv[0]

    #print(tf.shape(critic_model.trainable_variables))
    #print(tf.shape(grad_buffer_c[0]))
    #print(tf.shape(critic_model.trainable_variables[0]))
    #updating the networks
    critic_optimizer.apply_gradients(zip(grad_buffer_c, critic_model.trainable_variables))
    #actor is having some size/shape issue
    actor_optimizer.apply_gradients(zip(grad_buffer_a, actor_model.trainable_variables))

    if (episode+1)%50==0:
        print("Episode {}: Avg Score: {}".format(episode+1, np.mean(scores[-100:])))


