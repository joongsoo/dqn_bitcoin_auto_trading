# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import dqn
from collections import deque
from env.train import Environment
from data_convert import encode_money, encode_coin_cnt
from sms_util import send_sms
import pickle
from multiprocessing import Pool
from functools import partial



# 출력 사이즈
hidden_size = 3

# 총 입력길이
data_dim = 6
sequence_length = 144
input_size = data_dim * sequence_length
output_size = 3


# nn param
learning_rate = 1e-5
batch_size = 25000
min_learn_size = int(batch_size * 1.5)
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
    return len(replay_buffer) > min_learn_size

def replay_train(mainDQN, targetDQN, train_batch, episode):
    #with Pool(4) as pool:
    #    result_data = pool.map(partial(get_from_idx, target_list=train_batch), range(6))

    #states = result_data[0]
    #moneys = result_data[1]
    #next_moneys = result_data[2]
    #actions = result_data[3]
    #rewards = result_data[4]
    #next_states = result_data[5]

    states = np.vstack([[x[0]] for x in train_batch])
    moneys = np.vstack([x[1] for x in train_batch])
    next_moneys = np.vstack([x[2] for x in train_batch])
    actions = np.array([x[3] for x in train_batch])
    rewards = np.array([x[4] for x in train_batch])
    next_states = np.vstack([[x[5]] for x in train_batch])
    X = states

    Q_target = rewards + dis * np.max(targetDQN.predict(next_states, next_moneys), axis=1)

    y = mainDQN.predict(states, moneys)
    if episode > 1:
        print(y)
    y[np.arange(len(X)), actions] = Q_target
    if episode > 1:
        print(y)

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

        copy_ops = get_copy_var_ops()
        sess.run(copy_ops)

        episode = last_episode
        max_episodes = 30000
        frame = 0

        while episode < max_episodes:
            e = 1. / ((episode / 20) + 1)

            die = False
            clear = False
            state, before_money, before_coin_cnt = env.reset()
            before_money = encode_money(before_money)
            before_coin_cnt = encode_coin_cnt(before_coin_cnt)

            while not die and not clear:
                # random action
                if np.random.rand(1) < e or not is_learn_start():
                    action = env.get_random_actions()
                else:
                    action = np.argmax(targetDQN.predict([state], [[before_money, before_coin_cnt]]))

                # one step (1minute)
                # TODO : 1minute -> 1hour
                current_step, now_money, next_state, next_money, next_coin_cnt, reward, die, clear, penalty = env.step(action)
                next_money = encode_money(next_money)
                next_coin_cnt = encode_coin_cnt(next_coin_cnt)

                if die or penalty:
                    reward = 0

                replay_buffer.append((state, [before_money, before_coin_cnt], [next_money, next_coin_cnt], action, reward, next_state))

                if len(replay_buffer) > MAX_BUFFER_SIZE:
                    replay_buffer.popleft()
                """
                if is_learn_start():
                    minibatch = random.sample(replay_buffer, batch_size)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                    print("{}-{} : action[ {} ] / loss[ {} ]  /  money[ {} ] ".format(episode, current_step, action, loss, now_money))

                    if current_step % TARGET_UPDATE_FREQUENCY == 0:
                        sess.run(copy_ops)
                        try:
                            mainDQN.save(episode)
                            targetDQN.save(episode)
                        except:
                            print("save file not found")

                state = next_state
                if is_learn_start():
                    frame += 1
                """
                state = next_state
                before_money = next_money
                before_coin_cnt = next_coin_cnt

            print("================  GAME OVER  ===================")
            print("episode(step) : {}({})".format(episode, current_step))
            print("최종 잔액 : {}".format(now_money))
            print("================================================")

            # one episode one traning
            """
            for _ in range(int(MAX_BUFFER_SIZE / batch_size)):
                minibatch = random.sample(replay_buffer, batch_size)
                loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                print("loss : {}".format(loss))

            sess.run(copy_ops)
            
            try:
                mainDQN.save(episode)
                targetDQN.save(episode)
            except:
                print("save file not found")
            """            
            if is_learn_start():
                sms_str = "[home]\nepisode(step) : {}({})\n".format(episode, current_step) \
                          + "balance : {}".format(now_money)

                send_sms(sms_str)

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


