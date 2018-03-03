import numpy as np
import tensorflow as tf
from keras import backend as K

from app.ddpg.ReplayBuffer import ReplayBuffer
from app.ddpg.ActorNetwork import ActorNetwork
from app.ddpg.CriticNetwork import CriticNetwork
from app.ddpg.OU import OU

OU = OU()  # Ornstein-Uhlenbeck Process


class DDPGAgent:
    def __init__(self, env, model_config, save_folder):
        print("DDPG Agent Started")
        np.random.seed(model_config.seed)
        self.env = env
        self.ddpg_config = model_config.ddpg
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
        state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape[0]
        self.actor = ActorNetwork(sess, save_folder, state_dim, self.action_dim, model_config.ddpg)
        self.critic = CriticNetwork(sess, save_folder, state_dim, self.action_dim, model_config.ddpg)
        self.buff = ReplayBuffer(model_config.ddpg["buffer_size"])
        self.epsilon = 1

    def save_weights(self, episode_n):
        self.actor.save_weights(episode_n, avg_train=self.env.train_stats.mean_rewards[-1], avg_test=self.env.test_stats.mean_rewards[-1])
        self.critic.save_weights(episode_n, avg_train=self.env.train_stats.mean_rewards[-1], avg_test=self.env.test_stats.mean_rewards[-1])

    def train_episode(self):
        return self._play_episode(train_mode=True)

    def test_episode(self):
        return self._play_episode(train_mode=False)

    def _play_episode(self, train_mode):
        observation = self.env.reset_train_mode(train_mode)
        done = False
        total_reward = 0
        while not done:
            action = self.actor.model.predict(observation)[0]
            noise = np.zeros(self.action_dim)
            noise[0] = train_mode * max(self.epsilon, 0.1) * OU.function(action[0], 0.0, 0, 0.3)  # 0.60 = theta
            action = action + noise

            new_observation, reward, done, info = self.env.step(action)
            self.buff.add(observation, action, reward, new_observation, done)  # Add replay buffer

            # Do the batch update
            batch = self.buff.getBatch(self.ddpg_config["batch_size"])
            states = np.asarray([e[0] for e in batch]).reshape(-1, self.env.features_number, self.env.window_size, 1)
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch]).reshape(-1, self.env.features_number, self.env.window_size, 1)
            dones = np.asarray([e[4] for e in batch])
            y_t = np.zeros_like(actions)

            actor_prediction = self.actor.target_model.predict(new_states)
            target_q_values = self.critic.target_model.predict([new_states, actor_prediction])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + self.ddpg_config["gamma"] * target_q_values[k]

            if train_mode:
                self._train_step(states, actions, y_t)

            total_reward += reward
            observation = new_observation
        return total_reward

    def _train_step(self, states, actions, y_t):
        self.epsilon -= 1.0 / self.ddpg_config["explore"]
        self.critic.model.train_on_batch([states, actions], y_t)
        a_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()
