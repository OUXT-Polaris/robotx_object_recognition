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

def dataload():
    global batch_size, x_train, y_train, x_test, y_test, input_shape
    # データのロード
    batch_size = 128
    from keras.datasets import mnist
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # データの縮小用
    # x_train, y_train, x_test, y_test = x_train[:1024], y_train[:1024], x_test[:1024], y_test[:1024]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    print(y_test.shape, 'test label shape')

def main():
    # dataload()
    # TODO データのロードのパス、categoricalにする、フルカラー画像の場合はこれでいいの？
    from OUXT_imageData import api
    data_format = K.image_data_format()
    data_shape = (128, 128)
    if data_format == 'channels_first':
        input_shape = (3, 128, 128)
    else:
        input_shape = (128, 128, 3)

    batch_size = 32
    num_classes = 4
    gen = api.get_OUXT_ImageGeneratorDir(
            folder='OUXT_imageData/dataset',
            shape=data_shape, 
            batch_size=batch_size, 
            data_format=data_format)

    # modelの構築
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])
    print('compiled')

    # 学習
    hd_orig = 'test_run_1.hdf5'
    hd = 'test_run_1_trained_in_rb.hdf5'
    import os
    if os.path.exists(hd):
        print('loading')
        model.load_weights(hd)
    else:
        model.fit_generator(gen, steps_per_epoch=100, epochs=10)

        # model.fit(x_train, y_train,
        #         batch_size=batch_size,
        #         epochs=1,
        #         verbose=1,
        #         validation_data=(x_test, y_test))

        # save
        model.save_weights(hd_orig)

    # score = model.evaluate(x_test, y_test, verbose=0)
    # score = model.evaluate_generator(gen, steps=16)
    # print('evaluate', score)
    import matplotlib.pyplot as plt
    import numpy as np
    np.set_printoptions(precision=3, suppress=True)
    for x, y in gen:
        y_pred = model.predict(x, batch_size=batch_size)
        print(y.shape, y_pred.shape)
        for i in range(x.shape[0]):
            print(y[i], y_pred[i])
            plt.imshow(x[i])
            # plt.title('pred:{}(conf: {}) true:{}'.format(np.argmax(y_pred[i]), np.max(y_pred[i]), np.argmax(y[i])))
            plt.show()

    if True:
        return 0

    # エクスポートするよ
    print('export session')
    ksess = K.get_session()
    K.set_learning_phase(0)
    graph = ksess.graph
    kgraph = graph.as_graph_def()
    # print(kgraph)

    num_output = 1
    prefix = "output"
    pred = [None]*num_output
    outputName = [None]*num_output
    for i in range(num_output):
        outputName[i] = prefix + str(i)
        pred[i] = tf.identity(model.get_output_at(i), name=outputName[i])
    print('output name: ', outputName)

    constant_graph = graph_util.convert_variables_to_constants(
            ksess, 
            ksess.graph.as_graph_def(), 
            outputName)

    filename = "keras_mnist_cnn.pb"
    graph_io.write_graph(constant_graph, "./", filename, as_text=False)
    print('saved in', filename)


def use_from_tensorflow():
    # MNISTのデータを読み込む
    dataload()

    import tensorflow as tf
    filename = "keras_mnist_cnn.pb"
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

    print('shape', x_test.shape)
    with tf.Session(graph=graph) as sess:
        results = sess.run(outLayer.outputs[0],
                {inLayer.outputs[0]: x_test,
                    learnPhase.outputs[0]: 0})

    print(results.shape)
    print(y_test.shape)

def port_model():
    import keras.backend.tensorflow_backend as KTF
    KTF.set_session(sess)

    saver = tf.train.Saver()
    saver.save(sess, "models/" + "model.ckpt")
    tf.train.write_graph(sess.graph.as_graph_def(), "models/", "graph.pb")

if __name__ == '__main__':
    main()
    # use_from_tensorflow()