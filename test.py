# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:21:15 2018

@author: hugob
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

import seq2seq
from seq2seq.models import Seq2Seq