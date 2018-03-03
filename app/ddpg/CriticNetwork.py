import os
from keras.layers import Dense, Input, Conv1D, Conv2D, Flatten, add, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

WEIGHTS_FILE = "artifacts/base_critic_model.h5"


class CriticNetwork(object):
    def __init__(self, sess, save_folder, state_size, action_size, ddpg_config, model_weights=WEIGHTS_FILE):
        self.sess = sess
        self.model_weights = model_weights
        self.batch_size = ddpg_config["batch_size"]
        self.tau = ddpg_config["tau"]
        self.learning_rate = ddpg_config["learning_rate_actor"]
        self.action_size = action_size
        self.save_folder = save_folder
        K.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())
        self.load_weights()

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        kernel_initializer = tf.random_uniform_initializer(-3e-3, 3e-3)
        state_input = Input(shape=state_size)
        conv_1 = Conv2D(filters=128, kernel_size=[2, 7], strides=3, activation='relu', padding='valid')(state_input)
        conv_1 = BatchNormalization()(conv_1)
        conv_2 = Conv2D(filters=128, kernel_size=[1, 5], strides=1, activation='relu', padding='valid')(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_3 = Conv2D(filters=128, kernel_size=[1, 3], strides=1, activation='relu', padding='valid')(conv_2)
        conv_3 = BatchNormalization()(conv_3)
        flatten = Flatten()(conv_3)

        action_input = Input(shape=[action_dim], name='action2')
        a1 = Dense(256, activation='linear')(action_input)

        dense_1 = add([flatten, a1])
        dense_2 = Dense(256, activation='relu')(dense_1)

        output = Dense(action_dim, activation='linear', kernel_initializer=kernel_initializer)(dense_2)
        model = Model(inputs=[state_input, action_input], outputs=output)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, action_input, state_input

    def load_weights(self):
        try:
            self.model.load_weights(self.model_weights)
            self.target_model.load_weights(self.model_weights)
            print("Weights loaded successfully")
        except:
            print("Cannot find critic weights file")

    def save_weights(self, episode_n, avg_train, avg_test):
        filename = os.path.join(self.save_folder, "critic_model_%d_%.3f_%.3f.h5" % (episode_n, avg_train, avg_test))
        self.model.save_weights(filename)
