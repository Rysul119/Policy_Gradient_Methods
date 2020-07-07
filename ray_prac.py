import ray
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ray.init(num_cpus=4, num_gpus=1)

@ray.remote(num_gpus=0.25)
def f(env_name):
    import time
    # importing frameworks/libraries
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Lambda
    from tensorflow.keras.models import Model
    import pandas as pd
    import matplotlib.pyplot as plt
    import gym
    import numpy as np
    from timeit import default_timer as timer

    import pybulletgym
    import pybullet_envs
    '''gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])'''
    tf.keras.backend.set_floatx('float64')
    # declaring constants
    gamma = 0.99
    upds = [64, 256, 1024]
    lamb = 0.95
    # max_episodes = 3000
    tot_steps = 1000000
    env = gym.make(env_name)
    env.seed(15)
    np.random.seed(24)
    tf.random.set_seed(34)
    node = 64
    policy_learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
    mult_lrs = [1, 2]

    # getting the dimension of the environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    # act_bound_low = env.action_space.low[0]
    # print(act_bound, act_bound_low)
    std_bound = [1e-2, 1.0]

    for upd in upds:
        for policy_lr in policy_learning_rates:
            for mult_lr in mult_lrs:
                # models
                # policy model
                state_input = Input((state_dim,))
                dense_1 = Dense(node, activation='relu')(state_input)
                dense_2 = Dense(node, activation='relu')(dense_1)
                out_mu = Dense(action_dim, activation='tanh')(dense_2)
                mu_output = Lambda(lambda x: x * action_bound)(out_mu)
                std_output = Dense(action_dim, activation='softplus')(dense_2)
                policy_model = tf.keras.models.Model(state_input, [mu_output, std_output])
                policy_model_optimize = tf.keras.optimizers.Adam(policy_lr)

                # value model
                value_model = tf.keras.Sequential([
                    Input((state_dim,)),
                    Dense(node, activation='relu'),
                    Dense(node, activation='relu'),
                    Dense(1, activation='linear')
                ])
                value_lr = mult_lr * policy_lr
                value_model_optimizer = tf.keras.optimizers.Adam(value_lr)

                ep_score = 0
                ep_step = 0
                episode = 0
                scores = []
                steps = []

                state_batch = []
                action_batch = []
                td_target_batch = []
                advantage_batch = []
                done = False

                state = env.reset()

                print(
                    "Experiment is starting to learn {} using A2C with a policy learning rate {} and value learning rate {} and upd {}.".format(
                        env_name, policy_lr, value_lr, upd))
                start = timer()

                for step in range(tot_steps):
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
                            states_arr = np.append(states_arr, elem, axis=0)

                        actions_arr = action_batch[0]
                        for elem in action_batch[1:]:
                            actions_arr = np.append(actions_arr, elem, axis=0)

                        td_targets_arr = td_target_batch[0]
                        for elem in td_target_batch[1:]:
                            td_targets_arr = np.append(td_targets_arr, elem, axis=0)

                        advantages_arr = advantage_batch[0]
                        for elem in advantage_batch[1:]:
                            advantages_arr = np.append(advantages_arr, elem, axis=0)

                        # gradient and loss calculation
                        # policy
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

                        # value
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

                    ep_score += reward[0][0]
                    ep_step += 1
                    state = next_state[0]

                    if done:
                        scores.append(ep_score)
                        steps.append(ep_step)
                        episode += 1
                        print("Episode: {} Step: {} Reward: {}".format(episode, step, ep_score))
                        ep_score = 0
                        ep_step = 0
                        state = env.reset()

                end = timer()

                print(
                    'Experiment is finished to learn {} using A2C with a policy learning rate {} and value learning rate {} and upd {} after {} hours.'.format(
                        env_name, policy_lr, value_lr, upd, (end - start) / 3600.0))

                dict = {'r': scores, 'l': steps}
                df = pd.DataFrame(dict)
                logdir = 'logs/a2c/envs/' + env_name + '/lrs/'
                fname = 'plr=' + str(policy_lr) + 'mult=' + str(mult_lr) + 'upd=' + str(upd) + '.csv'
                os.makedirs(logdir, exist_ok=True)
                df.to_csv(logdir + fname)

    for upd in upds:
        for policy_lr in policy_learning_rates:
            for mult_lr in mult_lrs:
                df = pd.read_csv(
                    'logs/a2c/envs/' + env_name + '/lrs/plr=' + str(policy_lr) + 'mult=' + str(mult_lr) + 'upd=' + str(
                        upd) + '.csv')
                x = np.cumsum(df['l'])
                plt.scatter(x, df['r'], s=5, label='plr=' + str(policy_lr) + 'mult=' + str(mult_lr), alpha=0.4)
                plt.legend()

    plt.xlabel('Steps')
    plt.ylabel('Episodic Score')
    plt.title('Experiment on ' + env_name)

    logdir = 'logs/a2c/envs/' + env_name + '/lrs/'
    fname = env_name + '_opt.png'
    plt.savefig(logdir + fname)

# The four tasks created here can execute concurrently.
#words = ['I', 'am', 'Rysul']
env_names = ['HopperBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'HumanoidBulletEnv-v0', 'Walker2DBulletEnv-v0']
ray.get([f.remote(env_name) for env_name in env_names])
