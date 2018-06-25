#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reference:
https://qiita.com/yukiB/items/1ea109eceda59b26cd64#4-kerastensorflow%E3%81%A7%E4%BD%9C%E6%88%90%E3%81%97%E3%81%9F%E3%83%A2%E3%83%87%E3%83%AB%E3%82%92c%E3%81%8B%E3%82%89%E5%AE%9F%E8%A1%8C
"""
import tensorflow as tf
from tensorflow.python.framework import graph_util, graph_io
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras import backend as K
import numpy as np
import cv2


def use_from_tensorflow():

    filename = 'keras_object_recog.pb'
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    print(graph.get_operations())

    inLayer    = graph.get_operation_by_name('import/conv2d_1_input')
    learnPhase = graph.get_operation_by_name('import/dropout_1/keras_learning_phase')
    outLayer   = graph.get_operation_by_name('import/output0')

    # classes=['green', 'other', 'red', 'white']

    print("TEST")
    np.set_printoptions(precision=3, suppress=True)
    with tf.Session(graph=graph) as sess:
        # im = cv2.imread('../datasets/main/green/00002.png')
        # im = cv2.imread('../datasets/main/other/hoge.png')
        # im = cv2.imread('../datasets/main/red/00028.png')
        im = cv2.imread('../datasets/main/white/00011.png')

        print(im.shape)
        cv2.imshow('im',im)
        cv2.waitKey(3)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im / 255.
        im = cv2.resize(im, (128,128))
        im = im.reshape((1,128,128,3))

        y_pred = sess.run(outLayer.outputs[0],
                          {inLayer.outputs[0]: im,
                           learnPhase.outputs[0]: 0})

        print(y_pred)

        # for x, y in gen:
        #     print x[0].shape
        #     print
        #     im = x[0]
        #     im = im.reshape((128,128,3))
            
        #     cv2.imshow('im',cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(3)

        #     y_pred = sess.run(outLayer.outputs[0],
        #             {inLayer.outputs[0]: x,
        #                 learnPhase.outputs[0]: 0})
        #     print(y_pred, y)
        #     N += 1
        #     if N > 10:
        #         break

if __name__ == '__main__':
    use_from_tensorflow()
