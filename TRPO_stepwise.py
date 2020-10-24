import gym
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
from timeit import default_timer as timer
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

tf.keras.backend.set_floatx('float64')

# constants
gamma = 0.98
cg_iters = 10
cg_damping = 0.001
ent_coeff = 0.0
residual_tol = 1e-5
delta = 0.01
epsilon = 0.4
sample_iter = 1
backtrack_coeff = 0.6
backtrack_iters = 10
batch_size = 4096
# episodes =  1000
tot_steps = 1000000
lrs = [0.01, 0.001, 0.0001]
runs = 20

for lr in lrs:
    for run in range(runs):
        start = timer()
        print('Experiment is starting for learning rate {} and run {}'.format(lr, run + 21))

        env = gym.make('CartPole-v1')

        env.seed(19 + run + 20)
        np.random.seed(15 + run + 20)

        # print(env.action_space.n)

        # policy model
        policy_model = tf.keras.Sequential()
        policy_model.add(tf.keras.layers.Dense(64, input_shape=(env.observation_space.shape[0],), activation='relu'))
        # policy_model.add(tf.keras.layers.Dense(64, activation = 'relu'))
        policy_model.add(tf.keras.layers.Dense(env.action_space.n, activation='softmax'))

        policy_tmp_model = tf.keras.models.clone_model(policy_model)

        # value model
        value_model = tf.keras.Sequential()
        value_model.add(tf.keras.layers.Dense(64, input_shape=(env.observation_space.shape[0],), activation='relu'))
        # value_model.add(tf.keras.layers.Dense(64, activation = 'relu'))
        value_model.add(tf.keras.layers.Dense(1, activation='linear'))
        # add optimizer
        value_model_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # value_model.compile(value_model_optimizer, "mse")


        ep_score = 0
        ep_step = 0
        episode = 0
        scores = []
        steps = []
        upd_t = 0

        obs_all, actions_all, rs_all, action_probs_all, Gs_all = [None] * sample_iter, [None] * sample_iter, [
            None] * sample_iter, [None] * sample_iter, [None] * sample_iter
        mean_total_reward = [None] * sample_iter
        mean_entropy = [None] * sample_iter
        entropy = 0
        obs, actions, rs, action_probs, Gs = [], [], [], [], []
        ob = env.reset()
        done = False

        for step in range(tot_steps):

            ob = ob[np.newaxis, :]
            # ob = ob.reshape([4, 1])
            # print(ob)
            action_prob = policy_model(ob).numpy().ravel()
            # print(action_prob)
            # action = np.random.choice(action_prob.shape[0], p = action_prob)
            # Add epsilon greedy if above action choice does not work
            if np.random.uniform(0, 1) < epsilon:
                # print("Random action taken. epsilon: {}".format(epsilon))
                action = np.random.randint(0, env.action_space.n)
            else:
                # print("Biased random action taken.")
                action = np.random.choice(range(env.action_space.n), p=action_prob)  # act_prob.numpy()[0]
            new_ob, r, done, _ = env.step(action)
            rs.append(r)
            obs.append(ob[0])
            actions.append(action)
            action_probs.append(action_prob)
            entropy += -tf.reduce_sum(action_prob * tf.math.log(action_prob))
            ob = new_ob
            ep_step += 1
            if done:
                episode += 1
                steps.append(ep_step)
                ep_step = 0

                G = 0
                for r in rs[::-1]:
                    G = r + gamma * G
                    Gs.insert(0, G)

                mean_total_reward[0] = np.sum(rs)
                mean_entropy[0] = entropy / len(actions)
                obs_all[0] = obs
                actions_all[0] = actions
                action_probs_all[0] = action_probs
                Gs_all[0] = Gs
                epsilon = epsilon - 5e-3
                # value_loss = history.history["loss"][-1]

                if done:
                    mean_entropy = np.mean(mean_entropy)
                    mean_total_reward = np.mean(mean_total_reward)
                    best_reward = np.mean(mean_total_reward)
                    # print(obs_all)
                    obs_all = np.concatenate(obs_all)
                    # print(obs_all)
                    actions_all = np.concatenate(actions_all)
                    action_probs_all = np.concatenate(action_probs_all)
                    Gs_all = np.concatenate(Gs_all)

                    nbatches = len(obs_all) // batch_size
                    if len(obs_all) < batch_size:
                        nbatches += 1

                    # print(nbatches)
                    for batch_id in range(nbatches):
                        ob_batch = obs_all[batch_id * batch_size: (batch_id + 1) * batch_size]  # access the certain data 2D
                        # print(len(ob_batch))
                        Gs_batch = Gs_all[batch_id * batch_size: (batch_id + 1) * batch_size]
                        action_batch = actions_all[batch_id * batch_size: (batch_id + 1) * batch_size]
                        action_probs_batch = action_probs_all[
                                             batch_id * batch_size: (batch_id + 1) * batch_size]  ## access the certain data 2D
                        # print(len(action_probs_batch))
                        Vs = value_model(ob_batch).numpy().flatten()
                        advantage = np.array(Gs_batch) - Vs
                        # print(advantage)
                        actions_onehot = tf.one_hot(action_batch, env.action_space.n, dtype="float64")

                        # policy grad calculation
                        with tf.GradientTape() as policy_tape:
                            action_probs = policy_model(ob_batch)  # change it to not copy later
                            # print(ob_batch)
                            # print(action_probs)
                            action_prob = tf.reduce_sum(actions_onehot * action_probs, axis=1)
                            action_probs_old = policy_model(ob_batch)
                            action_prob_old = tf.reduce_sum(actions_onehot * action_probs_old, axis=1).numpy() + 1e-8
                            prob_ratio = action_prob / action_prob_old
                            # print(prob_ratio)
                            # get the policy loss
                            policy_loss = tf.reduce_mean(prob_ratio * advantage) + ent_coeff * mean_entropy
                        policy_grads = policy_tape.gradient(policy_loss, policy_model.trainable_variables,
                                                            unconnected_gradients=tf.UnconnectedGradients.ZERO)
                        policy_grads_flat = tf.concat([tf.reshape(g, [-1]) for g in policy_grads], axis=0)
                        policy_grads_flat = policy_grads_flat.numpy()
                        # print(policy_grads_flat.shape)

                        # conjugate grad
                        b = policy_grads_flat
                        x = np.zeros_like(b)
                        r = b.copy()
                        p = r.copy()
                        old_p = p.copy()
                        r_dot_old = np.dot(r, r)
                        for _ in range(cg_iters):
                            with tf.GradientTape() as fisher_tape:
                                with tf.GradientTape() as kl_tape:
                                    action_probs = policy_model(ob_batch)
                                    # action_prob = tf.reduce_sum(action_onehot * action_probs, axis = 1) + 1e-8
                                    action_probs_old = policy_model(ob_batch)
                                    # action_prob_old = tf.reduce_sum(action_onehot * action_probs, axis = 1)
                                    kl_div = tf.reduce_mean(
                                        tf.reduce_sum(action_probs_old * tf.math.log(action_probs_old / action_probs), axis=1))
                                kl_grad = kl_tape.gradient(kl_div, policy_model.trainable_variables,
                                                           unconnected_gradients=tf.UnconnectedGradients.ZERO)
                                kl_grad_flat = tf.concat([tf.reshape(g, [-1]) for g in kl_grad], axis=0)
                                kl_grad_flat = kl_grad_flat.numpy()  ## for this we were getting none in gradients
                                # print(kl_grad_flat.shape)
                                kl_grad_prod = tf.reduce_sum(kl_grad_flat * policy_grads_flat)
                                # print(kl_grad_prod)
                            fisher_vector_prod = fisher_tape.gradient(kl_grad_prod, policy_model.trainable_variables,
                                                                      unconnected_gradients=tf.UnconnectedGradients.ZERO)
                            # print(fisher_vector_prod)
                            fisher_vector_prod_flat = tf.concat([tf.reshape(g, [-1]) for g in fisher_vector_prod], axis=0)
                            # print(fisher_vector_prod_flat.shape)
                            z = fisher_vector_prod_flat.numpy() + (cg_damping * p)
                            alpha = r_dot_old / (np.dot(p, z) + 1e-8)
                            old_x = x
                            x += alpha * p
                            r -= alpha * z
                            r_dot_new = np.dot(r, r)
                            beta = r_dot_new / (r_dot_old + 1e-8)
                            r_dot_old = r_dot_new
                            if r_dot_old < residual_tol:
                                break
                            old_p = p.copy()
                            p = r + beta * p
                            if np.isnan(x).any():
                                print("x is nan")
                                print("z", np.isnan(z))
                                print("old_x", np.isnan(old_x))
                                print("kl_fn", np.isnan(kl_div))

                        step_direction = x
                        # GradientTape.gradient can only be called once on non-persistent tapes. below<< fixed
                        with tf.GradientTape() as fisher_tape_shs:
                            with tf.GradientTape() as kl_tape_shs:
                                action_probs = policy_model(ob_batch)
                                # action_prob = tf.reduce_sum(action_onehot * action_probs, axis = 1) + 1e-8
                                action_probs_old = policy_model(ob_batch)
                                # action_prob_old = tf.reduce_sum(action_onehot * action_probs, axis = 1)
                                kl_div = tf.reduce_mean(
                                    tf.reduce_sum(action_probs_old * tf.math.log(action_probs_old / action_probs), axis=1))
                            kl_grad_shs = kl_tape_shs.gradient(kl_div, policy_model.trainable_variables,
                                                               unconnected_gradients=tf.UnconnectedGradients.ZERO)
                            kl_grad_flat_shs = tf.concat([tf.reshape(g, [-1]) for g in kl_grad_shs], axis=0)
                            kl_grad_flat_shs = kl_grad_flat_shs.numpy()  ## for this we were getting none in gradients
                            # print(kl_grad_flat.shape)
                            kl_grad_prod_shs = tf.reduce_sum(kl_grad_flat_shs * step_direction)
                            # print(kl_grad_prod)
                        fisher_vector_prod_shs = fisher_tape_shs.gradient(kl_grad_prod_shs, policy_model.trainable_variables,
                                                                          unconnected_gradients=tf.UnconnectedGradients.ZERO)
                        # print(fisher_vector_prod_shs)
                        fisher_vector_prod_flat_shs = tf.concat([tf.reshape(g, [-1]) for g in fisher_vector_prod_shs], axis=0)
                        z = fisher_vector_prod_flat_shs.numpy() + (cg_damping * step_direction)
                        shs = 0.5 * step_direction.dot(z.T)
                        lm = np.sqrt(shs / delta) + 1e-8
                        fullstep = step_direction / lm

                        if np.isnan(fullstep).any():
                            print("fullstep is nan")
                            print("lm", lm)
                            print("step_direction", step_direction)
                            print("policy_gradient", policy_grads_flat)

                        oldtheta = tf.concat([tf.reshape(v, [-1]) for v in policy_model.trainable_variables], axis=0)
                        oldtheta = oldtheta.numpy()

                        for (n_backtracks, stepfrac) in enumerate(backtrack_coeff ** np.arange(backtrack_iters)):
                            x_new = oldtheta + stepfrac * fullstep
                            # assign vars for policy model copy
                            shapes = [v.shape.as_list() for v in policy_tmp_model.trainable_variables]
                            size_theta = np.sum([np.prod(shape) for shape in shapes])
                            start = 0
                            for i, shape in enumerate(shapes):
                                size = np.prod(shape)
                                param = tf.reshape(x_new[start:start + size], shape)  # change oldtheta to x_new
                                policy_tmp_model.trainable_variables[i].assign(param)
                                start += size
                            action_probs = policy_tmp_model(ob_batch)
                            action_prob = tf.reduce_sum(actions_onehot * action_probs, axis=1)
                            action_probs_old = policy_model(ob_batch)
                            action_prob_old = tf.reduce_sum(actions_onehot * action_probs_old, axis=1) + 1e-8
                            prob_ratio = action_prob / action_prob_old
                            # get the policy loss
                            newfval = tf.reduce_mean(prob_ratio * advantage) + ent_coeff * mean_entropy
                            # calculate kl_div
                            action_probs_kl = policy_tmp_model(ob_batch)
                            action_probs_old_kl = policy_model(ob_batch)
                            kl_div = tf.reduce_mean(
                                tf.reduce_sum(action_probs_old * tf.math.log(action_probs_old_kl / action_probs_kl), axis=1))
                            # debugging
                            if np.isnan(kl_div):
                                print("kl is nan")
                                print("xnew", np.isnan(x_new))
                                print("x", np.isnan(x))
                                print("stepfrac", np.isnan(stepfrac))
                                print("fullstep", np.isnan(fullstep))
                            if kl_div <= delta and newfval >= 0:
                                print("Linesearch worked at ", n_backtracks)
                                oldtheta = x_new
                                break
                            if n_backtracks == backtrack_iters - 1:
                                print("Lineasearch failed.", kl_div, newfval)
                        theta = oldtheta

                        # assign vars for policy model
                        shapes = [v.shape.as_list() for v in policy_model.trainable_variables]
                        size_theta = np.sum([np.prod(shape) for shape in shapes])
                        start = 0
                        for i, shape in enumerate(shapes):
                            size = np.prod(shape)
                            param = tf.reshape(theta[start:start + size], shape)
                            policy_model.trainable_variables[i].assign(param)
                            start += size
                        # define the surrogate loss again for this x value using policy model and policy old model(new theta). Maybe previous one will be both with policy model
                        # ^^done
                        # value function update
                        # Fix this part
                        # ^^ Fixed the dimensionality problem of v and Gs.
                        with tf.GradientTape() as value_tape:
                            v = value_model(ob_batch)
                            # Gs = np.array(Gs)
                            v = tf.reshape(v, [-1])
                            # print(Gs)
                            # print(v)
                            # print(tf.square(Gs - v))
                            value_loss = tf.reduce_mean(tf.square(Gs_batch - v)) * 0.5
                            # print("Value loss: {}".format(value_loss))
                        grads = value_tape.gradient(value_loss, value_model.trainable_variables)
                        # print(value_loss)
                        value_model_optimizer.apply_gradients(zip(grads, value_model.trainable_variables))
                        # if the plot is not good, use fit with the same lr.
                        # history = value_model.fit(ob_batch, Gs_batch, epochs = 5, verbose = 0)
                        scores.append(mean_total_reward)  # << check if indexing is required
                        print("Episode: {}, Total reward: {}, Policy loss: {}, Value loss: {}".format(episode, mean_total_reward, policy_loss, value_loss))

                        obs_all, actions_all, rs_all, action_probs_all, Gs_all = [None] * sample_iter, [None] * sample_iter, [None] * sample_iter, [None] * sample_iter, [None] * sample_iter
                        mean_total_reward = [None] * sample_iter
                        mean_entropy = [None] * sample_iter

                entropy = 0
                obs, actions, rs, action_probs, Gs = [], [], [], [], []
                ob = env.reset()
                done = False

        end = timer()
        dict = {'r': scores, 'l': steps}
        df = pd.DataFrame(dict)
        # save logged rewards and steps in the log directory
        log_dir = 'logs/CartPole-v1/TRPO/learning_rates/lr=' + str(lr) + '-' + str(run + 21)
        os.makedirs(log_dir, exist_ok=True)
        file_name = 'monitor.csv'
        df.to_csv(log_dir + '/' + file_name)
        print('Experiment is finished for learning rate {} and run {} which required {} hours.'.format(lr, run + 21, (end - start)/ 3600.0))

x = np.cumsum(df['l'])
plt.scatter(x, df['l'], s=5)
plt.xlabel('steps')
plt.ylabel('Average score')
plt.show()