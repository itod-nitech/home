# -*- coding: utf-8 -*-
#-----------------------------------------------------------
# MNIST文字認識 
# 書き方は/tensorflow/examples/tutorials/mnist/fully_connected_feed.pyを参考にしている
# 入力層->隠れ層->softmax層
#
# [使い方と注意点]
#
# ・python mnist3.py --help とすると利用できるすべてのオプションが表示される。
#
# ・トレーニングは、python mnist3.py --train=True (ver0.8では=Trueは必要なし)
#
# ・グラフプロットは、python mnist3.py --eval=True
#
# ・実験フェーズでは、スクリプトはなるべく書き換えないほうがよい（結果の再現性を保つのが困難になる）。
#
# ・変更が可能なパラメータは、オプションとして、プログラムに渡す
#
# ・実験ごとに train_dirを切り替えて実験結果を残す。
#
# ・プログラムの改良は、必要箇所をコメントアウトし試行錯誤するよりも新しいオプションスイッチを導入して、
#　オプションで切り替えるほうがよい。
#
# ・本プログラムでは、inference(),loss(),training()をこのスクリプト自身に含めているが、他スクリプトファイルに
# まとめるほうがよい場合もある（複数のアーキテクチャの比較検討の場合など）
#
# ・ログに数値を書き出す場合は、あとでpandsで読みやすい形式のCSVにしておくと次の処理が非常に楽になる。
#
#-----------------------------------------------------------

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns           
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import sys
import os
import time

#-----------------------------------------------------------
# データの入力
#-----------------------------------------------------------

import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True) # データ読み込み

#-----------------------------------------------------------
# 基礎パラメータの設定 (オプションフラグ関連の設定)
#-----------------------------------------------------------

flags = tf.app.flags
FLAGS = flags.FLAGS

# 基礎パラメータ群
flags.DEFINE_integer('H_units', 1000, 'Number of hidden units.')
flags.DEFINE_integer('batch_size', 100, 'Size of mini-batch.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_integer('max_steps', 500, 'Number of steps to run trainer.')
flags.DEFINE_string('train_dir', 'mnist3', 'Directory to put the training data.')

# 処理スイッチ群
flags.DEFINE_boolean('train', False, 'If True, traning process starts.')
flags.DEFINE_boolean('eval', False, 'If True, eval process starts.')
flags.DEFINE_boolean('plot', False, 'If True, plot figurs.')

H_units = FLAGS.H_units
batch_size = FLAGS.batch_size
learning_rate = FLAGS.learning_rate
max_steps = FLAGS.max_steps

#-----------------------------------------------------------
# 基礎パラメータのログ書き込み関数(オプションの形で出力)
#-----------------------------------------------------------

def write_params_to_log(fw):
    host=str(os.uname()[1])
    fw.write("This file is created by mnist3.py.\n") # 計算プログラムを明示する
    fw.write("%s\n"%datetime.now().strftime("%Y/%m/%d %H:%M:%S")) # 日付の表示
    fw.write("%s\n"%host) # 計算機環境
    fw.write("--H_units=%d "%H_units)
    fw.write("--batch_size=%d "%batch_size)
    fw.write("--learning_rate=%f "%learning_rate)
    fw.write("--max_steps=%d "%max_steps)
    fw.write("--train_dir=%s "%FLAGS.train_dir)
    fw.write("\n")

#-----------------------------------------------------------
# ネットワークの定義部
#-----------------------------------------------------------

w_h = tf.Variable(tf.random_normal([784, H_units], mean=0.0, stddev=0.05)) # 隠れ層の重み行列(ガウス乱数で初期化)
w_o = tf.Variable(tf.random_normal([H_units, 10], mean=0.0, stddev=0.05)) # 出力層の重み行列
b_h = tf.Variable(tf.zeros([H_units])) # 隠れ層バイアス
b_o = tf.Variable(tf.zeros([10]))  # 出力層バイアス
    
def inference(x):
    h = tf.nn.relu(tf.matmul(x, w_h) + b_h) # 隠れ層出力
    y = tf.nn.softmax(tf.matmul(h, w_o) + b_o) # 出力層出力 
    return y

def loss_function(y,y_):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    return cross_entropy

def training(loss):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) # 勾配法
    return train_step

def accuracy_measure(y,y_):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy
    
#-----------------------------------------------------------
# 訓練の実行
#-----------------------------------------------------------

def run_training():
    write_params_to_log(sys.stdout) # 標準出力に基礎パラメータを表示

    x = tf.placeholder("float", [None, 784]) # 入力層
    y_ = tf.placeholder("float", [None, 10]) # 出力層

    y = inference(x)
    loss = loss_function(y,y_)
    train_step = training(loss)
    accuracy = accuracy_measure(y,y_)
    
    saver = tf.train.Saver()
    init = tf.initialize_all_variables() 
    sess = tf.Session()    
    sess.run(init)
    
    log = open(FLAGS.train_dir + "/train_log","w") # 基礎パラメータとトレーニングの過程をログに残しておく
    write_params_to_log(log)
    start_time = time.time()
    print('step,\t loss,accuracy,\t  throughput(sec/step)')
    # pandasで扱いやすいようカラム名を書き込み(CSV、スペースやタブを入れないこと)
    log.write('step,loss,accuracy,throughput(sec/step)\n') 

    for step in range(1,max_steps):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _,loss_value = sess.run([train_step,loss], feed_dict={x: batch_xs, y_: batch_ys})

        if step % 100 == 0:
            duration = time.time() - start_time        
            accuracy_value = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
            print('%d,\t %.4f,\t %.4f,\t %.3f'% (step, loss_value, accuracy_value, duration/step))
            log.write('%d,\t %.4f,\t %.4f,\t %.3f\n' % (step, loss_value, accuracy_value, duration/step))

    log.close()
    saver.save(sess, FLAGS.train_dir+"/model") # 学習結果のセーブ

#-----------------------------------------------------------
#  テストデータを利用してテスト誤差を測定訓練の実行
#-----------------------------------------------------------
    
def eval_accuracy():

    x = tf.placeholder("float", [None, 784]) # 入力層
    y_ = tf.placeholder("float", [None, 10]) # 出力層

    y = inference(x)
    accuracy = accuracy_measure(y,y_)
    
    saver = tf.train.Saver()
    sess = tf.Session()    
    saver.restore(sess, FLAGS.train_dir+"/model") # 学習結果のリストア
    
    print('accuracy = ', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


#-----------------------------------------------------------
#  図の出力
#-----------------------------------------------------------

def plot_figs():
    f = pd.read_csv(FLAGS.train_dir+'/train_log',skiprows=4)
    print f

    f.plot(x='step',y='loss')
    plt.xlabel('Training step')
    plt.ylabel('Loss value')
    plt.yscale('log')
    plt.show()
    plt.savefig(FLAGS.train_dir+'/loss.pdf')
    plt.close()
    
    f.plot(x='step',y='accuracy')
    plt.xlabel('Training step')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig(FLAGS.train_dir+'/accuracy.pdf')
    plt.close()
    
    x = tf.placeholder("float", [None, 784]) # 入力層
    y = inference(x)
    saver = tf.train.Saver()
    init = tf.initialize_all_variables() 
    sess = tf.Session()    
    saver.restore(sess, FLAGS.train_dir+"/model") # 学習結果のリストア

    b_o_value = sess.run(b_o)
    plt.xlabel('b_o')
    plt.ylabel('Frequecy')
    sns.distplot(b_o_value, kde=False, rug=False, bins=30) 
    plt.show()
    plt.savefig(FLAGS.train_dir+'/b_o_.pdf')
    plt.close()

    w_o_value = sess.run(w_o)
    plt.xlabel('w_o')
    plt.ylabel('Frequecy')
    sns.distplot(w_o_value.reshape((1, -1)), kde=False, rug=False, bins=30) 
    plt.show()
    plt.savefig(FLAGS.train_dir+'/w_o_.pdf')
    plt.close()

    b_h_value = sess.run(b_h)
    plt.xlabel('b_h')
    plt.ylabel('Frequecy')
    sns.distplot(b_h_value, kde=False, rug=False, bins=30) 
    plt.show()
    plt.savefig(FLAGS.train_dir+'/b_h_.pdf')
    plt.close()

    w_h_value = sess.run(w_h)
    plt.xlabel('w_h')
    plt.ylabel('Frequecy')
    sns.distplot(w_h_value.reshape((1, -1)), kde=False, rug=False, bins=30) 
    plt.show()
    plt.savefig(FLAGS.train_dir+'/w_h_.pdf')
    plt.close()

#-----------------------------------------------------------
# main 関数 (フラグに基づくディスパッチ処理を行う)
#-----------------------------------------------------------
    
def main(_):
    if FLAGS.train==True:
        run_training()
    elif FLAGS.eval==True:
        eval_accuracy()
    elif FLAGS.plot==True:
        plot_figs()

        
if __name__ == '__main__':
    tf.app.run()
    
