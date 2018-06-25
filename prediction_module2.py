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


class Predicdion():

    def __init__(self):
        
        filename = 'keras_object_recog.pb'
        self.graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(filename, 'rb') as f:
            graph_def.ParseFromString(f.read())
        with self.graph.as_default():
            tf.import_graph_def(graph_def)

        self.inLayer    = self.graph.get_operation_by_name('import/conv2d_1_input')
        self.learnPhase = self.graph.get_operation_by_name('import/dropout_1/keras_learning_phase')
        self.outLayer   = self.graph.get_operation_by_name('import/output0')


    def __call__(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.
        image = cv2.resize(image, (128,128))
        image = image.reshape((1,128,128,3))

        with tf.Session(graph=self.graph) as sess:
            y_pred = sess.run(self.outLayer.outputs[0],
                              {self.inLayer.outputs[0]: image,
                               self.learnPhase.outputs[0]: 0})

        return y_pred


if __name__ == '__main__':

    # np.set_printoptions(precision=3, suppress=True)

    classes=['green', 'other', 'red', 'white']
    colors=[(152,251,152), (0,255,255), (203,192,255), (80,80,80)]

    prediction = Predicdion()

    im = cv2.imread('../datasets/main/green/00035.png')
    # im = cv2.imread('../datasets/main/other/hoge.png')
    # im = cv2.imread('../datasets/main/red/00028.png')
    # im = cv2.imread('../datasets/main/white/00011.png')

    print("TEST")
    y = prediction(im)
    index = np.argmax(y)
    print(classes[index])

    str_label = classes[index]

    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(im,str_label,(10,100),font, 1.5, colors[index],2)

    # print(im.shape)
    cv2.imshow('im',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
