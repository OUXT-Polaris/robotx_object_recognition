#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf

# test data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# variable作る関数 非ゼロを適当に入れておく
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# W:フィルタ x:入力 ストライドは1
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# max pooling function, zero padding
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# placeholder 後からfeeddictできる
x = tf.placeholder(tf.float32, [None, 784]) # 画像
y_ = tf.placeholder(tf.float32, [None, 10]) # 教師データ
x_image = tf.reshape(x, [-1, 28, 28, 1])  # shape = [28,28,1,1]

# 1st layer
W_conv1 = weight_variable([5,5,1,32])  # 5*5のフィルタを32種類とる
b_conv1 = bias_variable([32])  # 種類ごとの重み
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # shape = [28,28,1,32]
h_pool1 = max_pool_2x2(h_conv1)  # shape = [14,14,1,32]

# 2nd layer
W_conv2 = weight_variable([5,5,32,64])  # 5*5*32のサイズのフィルタを64種類
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # shape = [7,7,1,64]

# densely connected layer 
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  # shape = [1, 7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # shape = [1024] ベクトル


# dropout to avoid overfit
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # shape = [1024]

# readout
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  # shape = [10]


# 評価関数
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 計算グラフを定義
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)) # 正解かどうか and
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 計算実行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 学習
    for i in range(200):
        batch = mnist.train.next_batch(50)
        if i%2==0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_:batch[1], keep_prob:1.0})
            print('step %d : %g' % (i, train_accuracy))
        train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
    print('finish')
    print('test %g' % accuracy.eval(feed_dict={
        x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))



