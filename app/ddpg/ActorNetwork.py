import os
from keras.initializers import VarianceScaling
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Convolution2D, Conv2D, Concatenate, BatchNormalization
import tensorflow as tf
import keras.backend as K

WEIGHTS_FILE = "artifacts/base_actor_model.h5"


class ActorNetwork(object):
    def __init__(self, sess, save_folder, state_size, action_size, ddpg_config, model_weights=WEIGHTS_FILE):
        self.sess = sess
        self.model_weights = model_weights
        self.batch_size = ddpg_config["batch_size"]
        self.tau = ddpg_config["tau"]
        self.learning_rate = ddpg_config["learning_rate_actor"]
        self.action_size = action_size
        self.save_folder = save_folder
        K.set_session(sess)

        self.model, self.weights, self.state = self.create_actor_network(state_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())
        self.load_weights()

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size):
        kernel_initializer = tf.random_uniform_initializer(-3e-3, 3e-3)
        state_input = Input(shape=state_size)
        conv_1 = Conv2D(filters=128, kernel_size=[2, 7], strides=3, activation='relu', padding='valid')(state_input)
        conv_1 = BatchNormalization()(conv_1)
        conv_2 = Conv2D(filters=128, kernel_size=[1, 5], strides=1, activation='relu', padding='valid')(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_3 = Conv2D(filters=128, kernel_size=[1, 3], strides=1, activation='relu', padding='valid')(conv_2)
        conv_3 = BatchNormalization()(conv_3)
        flatten = Flatten()(conv_3)
        dense_1 = Dense(512, activation='relu')(flatten)
        dense_2 = Dense(256, activation='relu')(dense_1)
        output = Dense(self.action_size, activation='tanh', kernel_initializer=kernel_initializer)(dense_2)
        model = Model(inputs=state_input, outputs=output)
        return model, model.trainable_weights, state_input

    def load_weights(self):
        try:
            self.model.load_weights(self.model_weights)
            self.target_model.load_weights(self.model_weights)
            print("Weights loaded successfully")
        except:
            print("Cannot find actor weights file")

    def save_weights(self, episode_n, avg_train, avg_test):
        filename = os.path.join(self.save_folder, "actor_model_%d_%.3f_%.3f.h5" % (episode_n, avg_train, avg_test))
        self.model.save_weights(filename)
