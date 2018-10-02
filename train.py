# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import dqn
from collections import deque
from env.train import Environment
from data_convert import encode_money, encode_coin_cnt



# 출력 사이즈
hidden_size = 3

# 총 입력길이
data_dim = 6
sequence_length = 144
input_size = data_dim * sequence_length
output_size = 3


# nn param
learning_rate = 1e-4
batch_size = 100000
min_learn_size = int(batch_size * 1.5)
dis = 0.9 # 미래가중치


replay_buffer = deque()
MAX_BUFFER_SIZE = 1000000
TARGET_UPDATE_FREQUENCY = 10

def is_learn_start():
    return len(replay_buffer) > min_learn_size

def replay_train(mainDQN, targetDQN, train_batch):
    states = np.vstack([[x[0]] for x in train_batch])
    moneys = np.array([x[1] for x in train_batch])
    next_moneys = np.array([x[2] for x in train_batch])
    actions = np.array([x[3] for x in train_batch])
    rewards = np.array([x[4] for x in train_batch])
    next_states = np.vstack([[x[5]] for x in train_batch])

    X = states

    Q_target = rewards + dis * np.max(targetDQN.predict(next_states, next_moneys), axis=1)

    y = mainDQN.predict(states, moneys)
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
                current_step, before_money, before_coin_cnt, now_money, next_state, next_money, next_coin_cnt, reward, die, clear = env.step(action)
                before_money = encode_money(before_money)
                before_coin_cnt = encode_coin_cnt(before_coin_cnt)
                next_money = encode_money(next_money)
                next_coin_cnt = encode_coin_cnt(next_coin_cnt)


                if die:
                    reward = -10000

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

            print("================  GAME OVER  ===================")
            print("episode(step) : {}({})".format(episode, current_step))
            print("최종 잔액 : ", now_money)
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
                for _ in range(100):
                    minibatch = random.sample(replay_buffer, batch_size)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                    print("loss : {}".format(loss))

                sess.run(copy_ops)

                try:
                    mainDQN.save(episode)
                    targetDQN.save(episode)
                except:
                    print("save file not found")
                episode += 1




if __name__ == "__main__":
    main()


