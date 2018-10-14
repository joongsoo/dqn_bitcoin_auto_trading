# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import variance_scaling_initializer


class DQN:
    def __init__(self, session, seq_length, data_dim, output_size, l_rate, name="main"):
        self.session = session
        self.seq_length = seq_length
        self.data_dim = data_dim
        self.l_rate = l_rate

        self.output_size = output_size

        self.net_name = name

        self._build_network()
        self.saver = tf.train.Saver()
        self.save_path = "./save/save_model_" + self.net_name + ".ckpt"
        tf.logging.info(name + " - initialized")

    # Make a lstm cell with hidden_size (each unit output vector size)
    def lstm_cell(self, size, keep_prob):
        cell = tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True, activation=tf.tanh)
        drop = rnn.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
        return drop

    def _build_network(self):
        with tf.variable_scope(self.net_name):

            # input place holders
            self._X = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim], name="input_x")
            self._MONEY = tf.placeholder(tf.float32, [None, 3], name="input_money")
            self._keep_prob = tf.placeholder(tf.float32, name="kp")
            self._train_mode = tf.placeholder(tf.bool, name='train_mode')

            bn_params = {
                'is_training': self._train_mode,
                'decay': 0.9,
                'fused': False
            }

            multi_cells = rnn.MultiRNNCell([self.lstm_cell(self.data_dim, self._keep_prob) for _ in range(2)],
                                           state_is_tuple=True)

            outputs, _states = tf.nn.dynamic_rnn(multi_cells, self._X, dtype=tf.float32)

            rnn_output = fully_connected(outputs, self.data_dim, activation_fn=tf.nn.relu,
                                                           weights_initializer=variance_scaling_initializer(dtype=tf.float32),
                                                            normalizer_fn=batch_norm,
                                                            normalizer_params=bn_params)
            rnn_output = tf.nn.dropout(rnn_output, keep_prob=self._keep_prob)

            rnn_output = tf.reshape(rnn_output, [-1, self.seq_length * self.data_dim])

            money = fully_connected(self._MONEY, 128, activation_fn=tf.nn.relu,
                                                           weights_initializer=variance_scaling_initializer(dtype=tf.float32),
                                                            normalizer_fn=batch_norm,
                                                            normalizer_params=bn_params)
            money = tf.nn.dropout(money, keep_prob=self._keep_prob)

            output = fully_connected(rnn_output, 128, activation_fn=tf.nn.relu,
                                                           weights_initializer=variance_scaling_initializer(dtype=tf.float32),
                                                            normalizer_fn=batch_norm,
                                                            normalizer_params=bn_params)
            output = tf.nn.dropout(output, keep_prob=self._keep_prob)

            output = fully_connected(output + money, 256, activation_fn=tf.nn.relu,
                                                           weights_initializer=variance_scaling_initializer(dtype=tf.float32),
                                                            normalizer_fn=batch_norm,
                                                            normalizer_params=bn_params)
            output = tf.nn.dropout(output, keep_prob=self._keep_prob)

            output = fully_connected(output, 512, activation_fn=tf.nn.relu,
                                                           weights_initializer=variance_scaling_initializer(dtype=tf.float32),
                                                            normalizer_fn=batch_norm,
                                                            normalizer_params=bn_params)
            output = tf.nn.dropout(output, keep_prob=self._keep_prob)

        self._Qpred = fully_connected(output, self.output_size, activation_fn=tf.nn.relu,
                                                           weights_initializer=variance_scaling_initializer(dtype=tf.float32))

        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

        self._train = tf.train.AdamOptimizer(learning_rate=self.l_rate).minimize(self._loss)


    def save(self, episode=0):
        self.saver.save(self.session, self.save_path+ "-" + str(episode))

    def restore(self, episode=0):
        load_path = self.save_path + "-" + str(episode)
        self.saver.restore(self.session, load_path)

    def predict(self, state, money):
        predict = self.session.run(self._Qpred, feed_dict={self._X: state,
                                                           self._MONEY: money,
                                                           self._keep_prob: 1.0,
                                                           self._train_mode: False})
        return predict

    def update(self, x_stack, money_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={
            self._X: x_stack,
            self._MONEY: money_stack,
            self._Y: y_stack,
            self._keep_prob: 0.7,
            self._train_mode: True
        })

