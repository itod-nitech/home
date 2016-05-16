# -*- coding: utf-8 -*-
#-----------------------------------------------------------
# ２値圧縮センシングの再現アルゴリズム(スパース重ね合わせ符号)
# 書き換え版(伊藤:2016/5)
# [使い方と注意点]
# ・python mnist3.py --help とすると利用できるすべてのオプションが表示される。
# ・トレーニングは、python mnist3.py --train=True (ver0.8では=Trueは必要なし)
# ・変更が可能なパラメータは、オプションとして、プログラムに渡す
# ・実験ごとに train_dirを切り替えて実験結果を残す。
#-----------------------------------------------------------

import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns           
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import sys
import os
import time

#-----------------------------------------------------------
# 基礎パラメータの設定 (オプションフラグ関連の設定)
#-----------------------------------------------------------

flags = tf.app.flags
FLAGS = flags.FLAGS

# 基礎パラメータ群
flags.DEFINE_integer('n', 200, 'Length of sparse vector')
flags.DEFINE_integer('m', 100, 'Length of obsarvation vector')
