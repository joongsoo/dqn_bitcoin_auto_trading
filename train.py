# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import dqn
from collections import deque
from env.train import Environment
from data_convert import encode_money, encode_coin_cnt, encode_avg_price
from sms_util import send_sms
import pickle

# 출력 사이즈
hidden_size = 3

# 총 입력길이
data_dim = 6
sequence_length = 144
input_size = data_dim * sequence_length
output_size = 3


# nn param
learning_rate = 1e-3
batch_size = 70000
dis = 0.9 # 미래가중치

replay_buffer = deque()

try:
    with open("save/train_queue.pkl", "rb") as f:
        replay_buffer = pickle.load(f)
        print("train_queue loaded")
except FileNotFoundError:
    print("train_queue file not exists")

MAX_BUFFER_SIZE = 1000000
TARGET_UPDATE_FREQUENCY = 5

def get_from_idx(idx, target_list):
    return np.vstack([[item[idx]] for item in target_list])

def is_learn_start():
    return len(replay_buffer) >= MAX_BUFFER_SIZE

def replay_train(mainDQN, targetDQN, train_batch, episode):
    states = np.vstack([[x[0]] for x in train_batch])
    moneys = np.vstack([x[1] for x in train_batch])
    next_moneys = np.vstack([x[2] for x in train_batch])
    actions = np.array([x[3] for x in train_batch])
    rewards = np.array([x[4] for x in train_batch])
    next_states = np.vstack([[x[5]] for x in train_batch])
    finish = np.array([x[6] for x in train_batch])
    X = states

    Q_target = rewards + dis * np.max(targetDQN.predict(next_states, next_moneys), axis=1) * ~finish

    y = mainDQN.predict(states, moneys)

    # 액션 비율 로깅
    predict_actions = np.argmax(y, axis=1)
    unique, counts = np.unique(predict_actions, return_counts=True)
    print(dict(zip(unique, counts)))

    y[np.arange(len(X)), actions] = Q_target

    return mainDQN.update(X, moneys, y)


def get_copy_var_ops(dest_scope_name="target", src_scope_name="main"):
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def main():
    # init env
    start_money = 1000000
    env = Environment(start_money, sequence_length)

    # run
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, sequence_length, data_dim, output_size, learning_rate, name="main")
        targetDQN = dqn.DQN(sess, sequence_length, data_dim, output_size, learning_rate, name="target")
        tf.global_variables_initializer().run()

        last_episode = 0

        try:
            mainDQN.restore(last_episode)
            targetDQN.restore(last_episode)
        except:
            print("save file not found")

        last_episode += 1

        copy_ops = get_copy_var_ops()
        sess.run(copy_ops)

        episode = last_episode
        max_episodes = 30000

        while episode < max_episodes:
            e = 1. / ((episode / 20) + 1)

            die = False
            clear = False
            state, before_money, before_coin_cnt, before_avg_price = env.reset()
            before_money = encode_money(before_money)
            before_coin_cnt = encode_coin_cnt(before_coin_cnt)
            before_avg_price = encode_avg_price(before_avg_price)

            while not die and not clear:
                # random action
                if np.random.rand(1) < e or not is_learn_start():
                    action = env.get_random_actions()
                else:
                    action = np.argmax(targetDQN.predict([state], [[before_money, before_coin_cnt, before_avg_price]]))

                # one step (1minute)
                # TODO : 1minute -> 1hour
                current_step, now_money, next_state, next_money, next_coin_cnt, next_avg_buy_price, reward, die, clear = env.step(action)
                next_money = encode_money(next_money)
                next_coin_cnt = encode_coin_cnt(next_coin_cnt)
                next_avg_buy_price = encode_avg_price(next_avg_buy_price)

                if die:
                    reward = -100.

                replay_buffer.append((state, [before_money, before_coin_cnt, before_avg_price], [next_money, next_coin_cnt, next_avg_buy_price], action, reward, next_state, die))

                if len(replay_buffer) > MAX_BUFFER_SIZE:
                    replay_buffer.popleft()

                state = next_state
                before_money = next_money
                before_coin_cnt = next_coin_cnt
                before_avg_price = next_avg_buy_price

            print("================  GAME OVER  ===================")
            print("episode(step) : {}({})".format(episode, current_step))
            print("최종 잔액 : {}".format(now_money))
            print("================================================")

            if is_learn_start():
                sms_str = "[home]\nepisode(step) : {}({})\n".format(episode, current_step) \
                          + "balance : {}".format(now_money)

                try:
                    send_sms(sms_str)
                except:
                    pass

                with open("save/train_queue.pkl", "wb") as f:
                    pickle.dump(replay_buffer, f)

                for idx in range(100):
                    minibatch = random.sample(replay_buffer, batch_size)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch, episode)

                    if idx % TARGET_UPDATE_FREQUENCY == 0:
                        sess.run(copy_ops)

                    print("loss : {}".format(loss))

                try:
                    mainDQN.save(episode)
                    targetDQN.save(episode)
                except:
                    print("save file not found")

                episode += 1


if __name__ == "__main__":
    main()


