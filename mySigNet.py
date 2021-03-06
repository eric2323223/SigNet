# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:24:27 2017
@author: sounak_dey and anjan_dutta
"""

import numpy as np

np.random.seed(1337)  # for reproducibility
import os
import argparse

# from keras.utils.visualize_util import plot

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
# from keras.preprocessing import image
from keras import backend as K
import getpass as gp
import sys
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import random
import keras.preprocessing.image as img
from pathlib import Path
# import matplotlib.pyplot as plt
from PIL import Image
import itertools
from keras.models import model_from_json
# import matplotlib.image as mpimg

random.seed(1337)

# Create a session for running Ops on the Graph.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_base_network_signet(input_shape):
    seq = Sequential()
    seq.add(Convolution2D(96, 11, 11, activation='relu', name='conv1_1', subsample=(4, 4), input_shape=input_shape,
                          init='glorot_uniform', dim_ordering='tf'))
    seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(ZeroPadding2D((2, 2), dim_ordering='tf'))

    seq.add(Convolution2D(256, 5, 5, activation='relu', name='conv2_1', subsample=(1, 1), init='glorot_uniform',
                          dim_ordering='tf'))
    seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))  # added extra
    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))

    seq.add(Convolution2D(384, 3, 3, activation='relu', name='conv3_1', subsample=(1, 1), init='glorot_uniform',
                          dim_ordering='tf'))
    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))

    seq.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', subsample=(1, 1), init='glorot_uniform',
                          dim_ordering='tf'))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))  # added extra
    #    model.add(SpatialPyramidPooling([1, 2, 4]))
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(1024, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform'))
    seq.add(Dropout(0.5))

    seq.add(Dense(128, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform'))  # softmax changed to relu
    seq.summary()
    return seq


def compute_accuracy_roc(predictions, labels):
    '''Compute ROC accuracy with a range of thresholds on distances.
    '''
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)

    step = 0.01
    max_acc = 0

    for d in np.arange(dmin, dmax + step, step):
        idx1 = predictions.ravel() <= d
        idx2 = predictions.ravel() > d

        tpr = float(np.sum(labels[idx1] == 1)) / nsame
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff
        acc = 0.5 * (tpr + tnr)
        #       print ('ROC', acc, tpr, tnr)

        if (acc > max_acc):
            max_acc = acc

    return max_acc

def loadImage(img):
    img = Image.open(img)
    w, h = img.size
    margin = 30
    area = (margin, margin, w - margin, h - margin)
    cropped = img.crop(area)
    cropped = cropped.resize((300, 120), Image.ANTIALIAS)
    return np.array(cropped)[:,:,0:1]

def loadImages(path):
    imgs = os.listdir(path)
    dict = {}
    gfiles = []
    ffiles = []
    for img in imgs:
        if len(img)>9:
            gfiles.append(img)
        else:
            ffiles.append(img)
        imgfile = path+'/'+img
        imgArray = loadImage(imgfile)
        dict[img] = imgArray
    return gfiles, ffiles, dict


def preprocessImage(image):
    img = Image.open(image)
    w, h = img.size
    margin = 30
    area = (margin, margin, w - margin, h - margin)
    cropped = img.crop(area)
    return cropped.resize((300, 120), Image.ANTIALIAS)

def getFiles(path):
    files = os.listdir(path)
    return [list(v) for k, v in itertools.groupby(files, key=lambda x: x[:3])]

def genTruePairs(list):
    pairs = []
    for i in range(0, len(list)):
        for j in range(i + 1, len(list)):
            pairs.append([list[i], list[j]])
    return pairs

def genFalsePairs(glist, flist):
    pairs = []
    for i in range(0, len(glist)):
        for j in range(0, len(flist)):
            pairs.append([glist[i], flist[j]])
    return pairs

def getData2():
    X1 = []
    X2 = []
    Y = []

    path = 'c:/users/eric/workspace/DL_data/signatures'
    sigdirs = os.listdir(path)
    for dir in sigdirs:
        gfiles, ffiles, dict = loadImages(path+'/'+dir)
        truePairs = genTruePairs(gfiles)
        falsePairs = genFalsePairs(gfiles, ffiles)
        for pair in truePairs:
            X1.append(dict[pair[0]])
            X2.append(dict[pair[1]])
            Y.append(1)
        for pair in falsePairs:
            X1.append(dict[pair[0]])
            X2.append(dict[pair[1]])
            Y.append(0)
    return np.array(X1), np.array(X2), np.array(Y)


def getData():
    X1 = []
    X2 = []
    Y = []

    path1 = Path(__file__).resolve().parent.parent.parent.joinpath(
        "DL_data/trainingSet/OfflineSignatures/Offline Genuine")
    path2 = Path(__file__).resolve().parent.parent.parent.joinpath(
        "DL_data/trainingSet/OfflineSignatures/Offline forgeries")

    path = Path(__file__).resolve().parent.parent.parent.joinpath(
        "DL_data/trainingSet/OfflineSignatures/chinese/trainingSet")
    signatureDirs = os.listdir(path)
    for dir in signatureDirs:
        gSignatureFiles = os.listdir(Path.joinpath(path, dir, 'genuine'))
        fSignatureFiles = os.listdir(path.joinpath(path, dir, 'forgeries'))
        for pair in genTruePairs(gSignatureFiles):
            # print(pair)
            img1 = preprocessImage(Path.joinpath(path, dir, 'genuine', pair[0]))
            img2 = preprocessImage(Path.joinpath(path, dir, 'genuine', pair[1]))
            # img.show()
            # image = img.load_img(path, target_size=(120, 300))
            # x = img.img_to_array(cropped)
            x1 = np.array(img1)[:, :, 0:1]
            x2 = np.array(img2)[:, :, 0:1]
            # print(x.shape)
            X1.append(x1)
            X2.append(x2)
            Y.append(1)
        for pair in genFalsePairs(gSignatureFiles, fSignatureFiles):
            # print(pair)
            img1 = preprocessImage(Path.joinpath(path, dir, 'genuine', pair[0]))
            img2 = preprocessImage(Path.joinpath(path, dir, 'forgeries', pair[1]))
            # img.show()
            # image = img.load_img(path, target_size=(120, 300))
            # x = img.img_to_array(cropped)
            x1 = np.array(img1)[:, :, 0:1]
            x2 = np.array(img2)[:, :, 0:1]
            # print(x.shape)
            X1.append(x1)
            X2.append(x2)
            Y.append(0)

    # gfiles = getFiles(path1)
    # ffiles = getFiles(path2)
    #
    # for name in gfiles:
    #     for pair in genTruePairs(name):
    #         print(pair)
    #
    #         img1 = preprocessImage(Path.joinpath(path1,pair[0]))
    #         img2 = preprocessImage(Path.joinpath(path1,pair[1]))
    #         # img.show()
    #         # image = img.load_img(path, target_size=(120, 300))
    #         # x = img.img_to_array(cropped)
    #         x1 = np.array(img1)[:,:,0:1]
    #         x2 = np.array(img2)[:,:,0:1]
    #         # print(x.shape)
    #         X1.append(x1)
    #         X2.append(x2)
    #         Y.append(1)

    return np.array(X1), np.array(X2), np.array(Y)
    # imgplot = plt.imshow(x)
    # plt.show()


def train(saveModel=True):
    batch_sz = 128
    nsamples = 276
    # img_height = 155
    # img_width = 220
    img_height = 120
    img_width = 300
    # img_height = 360
    # img_width = 900

    featurewise_center = False
    featurewise_std_normalization = True
    zca_whitening = False
    nb_epoch = 20
    input_shape = (img_height, img_width, 1)

    # # initialize data generator
    # datagen = SignatureDataGenerator(dataset, tot_writers, num_train_writers,
    #     num_valid_writers, num_test_writers, nsamples, batch_sz, img_height, img_width,
    #     featurewise_center, featurewise_std_normalization, zca_whitening)

    # data fit for std
    # X_sample = read_signature_data(dataset, int(0.5*tot_writers), height=img_height, width=img_width)
    # datagen.fit(X_sample)
    # del X_sample

    # network definition
    base_network = create_base_network_signet(input_shape)

    input_a = Input(shape=(input_shape))
    input_b = Input(shape=(input_shape))

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)
    model.summary()

    # compile model
    rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
    adadelta = Adadelta()
    model.compile(loss=contrastive_loss, optimizer=rms)

    X1, X2, Y = getData2()
    print(len(Y))

    model.fit(x=[X1, X2], y=Y, epochs=8, validation_split=0.2)
    # serialize model to JSON

    if(saveModel == True):
        model_json = model.to_json()
        with open("mysignet.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("mysignet.h5")
        print("Saved model to disk")

    # display model

def predict(file1, file2):
    json_file = open('mysignet.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("mysignet.h5")
    print("Loaded model from disk")
    sig1 = loadImage(file1)
    sig2 = loadImage(file2)
    prediction = loaded_model.predict([np.array([sig1]), np.array([sig2])])
    print(prediction)


# Main Function
if __name__ == "__main__":

    # train()
    file1 = 'C:/Users/eric/workspace/DL_data/signatures/001/001_1.png'
    file2 = 'C:/Users/eric/workspace/DL_data/signatures/001/0113001_4.png'
    predict(file1, file2)
