# -*- coding: utf-8 -*-
#-----------------------------------------------------------
# ２値圧縮センシングの再現アルゴリズム(スパース重ね合わせ符号)
# シンプル版
#-----------------------------------------------------------

import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *

#-----------------------------------------------------------
# 基礎パラメータ
#-----------------------------------------------------------

n = 200                          # 疎ベクトル長
m = 100                          # 観測ベクトル長
H_units = 1000                          # 隠れ層のユニット数
#H_units = 2000

batch_size = 50                       # ミニバッチのサイズ
num_ones=5                      # 1の個数
learning_rate=0.02              # 学習率
#learning_rate=0.002
L1_reg=0.8                     # L1正則化項の係数
max_steps = 5000               # トレーニング回数

#-----------------------------------------------------------
# ミニバッチ生成
#-----------------------------------------------------------

def gen_sp_vector():            # 疎ベクトル生成
    ret = np.zeros(n)
    for i in range(num_ones):
        ret[randint(n)]=1
    return [ret];

def gen_batch():                # ミニバッチを返す
    batch_x=np.empty([0,n])
    for i in range(batch_size):
        x = gen_sp_vector()
        batch_x = np.r_[batch_x, x]
    return batch_x

#-----------------------------------------------------------
# ネットワークの定義部
#-----------------------------------------------------------

x = tf.placeholder("float", [None, n]) # 入力層(スパースベクトル)
A = tf.constant(randn(n,m),"float")          # 観測行列

w_h = tf.Variable(tf.random_normal([m, H_units], mean=0.0, stddev=0.05)) # 隠れ層の重み行列
b_h = tf.Variable(tf.zeros([H_units])) # 隠れ層バイアス
w_o = tf.Variable(tf.random_normal([H_units, n], mean=0.0, stddev=0.05)) # 出力層の重み行列
b_o = tf.Variable(tf.zeros([n]))  # 出力層バイアス

u = tf.sign(tf.matmul(x,A))                         # 観測ベクトルの生成

h = tf.nn.sigmoid(tf.matmul(u, w_h) + b_h) # 隠れ層出力
y = tf.nn.sigmoid(tf.matmul(h, w_o) + b_o) # 出力層出力

#-----------------------------------------------------------
# ロス関数定義部
#-----------------------------------------------------------

loss = -tf.reduce_mean(x*tf.log(y+1.e-8))+ L1_reg*tf.reduce_mean(tf.abs(y))

#-----------------------------------------------------------
# オプティマイザの指定
#-----------------------------------------------------------

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss) 

#-----------------------------------------------------------
# 訓練の実行
#-----------------------------------------------------------

init = tf.initialize_all_variables() # 変数の初期化

with tf.Session() as sess:
    sess.run(init)
    print('Training...')
    step_list=[]
    e_list=[]
    for step in range(max_steps):
        batch_xs = gen_batch()
        _,e = sess.run([train_step,loss], feed_dict={x: batch_xs})        
        if step % 100 == 0:
            print('  step, obj = %6d: %6.3f' % (step, e))
            step_list = step_list + [step]
            e_list = e_list + [e]
    xs = gen_sp_vector()
    xrep = y.eval(feed_dict={x:xs})

#-----------------------------------------------------------
# 結果のプロット
#-----------------------------------------------------------
    print A.eval()              # 観測行列の表示

    plt.subplot(3, 1, 1)
    plt.xlabel('number of iterations')
    plt.ylabel('squared error')
    plt.plot(step_list, e_list)    # ロス関数の推移

    plt.subplot(3, 1, 2)
    plt.plot(range(n),xs[0])    # 元の疎ベクトル

    plt.subplot(3, 1, 3)
    plt.plot(range(n),xrep[0])  # サポート推定結果

    plt.show()
    
