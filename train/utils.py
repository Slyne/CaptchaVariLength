#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import Convolution2D, MaxPooling2D, GRU, TimeDistributed
from keras.layers import Dense, Dropout, Activation, Flatten, RepeatVector,Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from keras import backend as K
K.set_image_dim_ordering("th")
from config import *

import random

def load_data(split_ratio):
    with open(images_dir, "rb") as f:
        images = np.load(f)
    with open(labels_dir, "rb") as f:
        labels = np.load(f)

    vocab_size = len(ch_index)
    labels_categorical = np.asarray([to_categorical(label, vocab_size) for label in labels])
    print "images shape", images.shape
    # print images[0]
    print "input labels shape", labels_categorical.shape
    total = images.shape[0]
    seed = range(total)
    random.shuffle(seed)
    split_index = int(total*split_ratio)
    train_data = images[seed[0:split_index]]
    train_label = labels_categorical[seed[0:split_index]]
    val_data = images[seed[split_index:]]
    val_label = labels_categorical[seed[split_index:]]
    return (train_data, train_label, val_data, val_label)


def create_simpleCnnRnn(image_shape, max_caption_len,vocab_size):
    image_model = Sequential()
    # image_shape : C,W,H
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    image_model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=image_shape))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(32, 3, 3))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))
    image_model.add(Dropout(0.25))
    image_model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(64, 3, 3))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))
    image_model.add(Dropout(0.25))
    image_model.add(Flatten())
    # Note: Keras does automatic shape inference.
    image_model.add(Dense(128))
    image_model.add(RepeatVector(max_caption_len))
    image_model.add(Bidirectional(GRU(output_dim=128, return_sequences=True)))
    #image_model.add(GRU(output_dim=128, return_sequences=True))
    image_model.add(TimeDistributed(Dense(vocab_size)))
    image_model.add(Activation('softmax'))
    return image_model


from seq2seq.models import AttentionSeq2Seq, Seq2Seq
def create_imgText(image_shape, max_caption_len,vocab_size):
    image_model = Sequential()
    # image_shape : C,W,H
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    image_model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=image_shape))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(32, 3, 3))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))
    image_model.add(Dropout(0.25))
    image_model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(64, 3, 3))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))
    image_model.add(Dropout(0.25))
    image_model.add(Flatten())
    # Note: Keras does automatic shape inference.
    image_model.add(Dense(128))
    image_model.add(RepeatVector(1))
    #model = AttentionSeq2Seq(input_dim=128, input_length=1, hidden_dim=128, output_length=max_caption_len, output_dim=vocab_size)
    model = Seq2Seq(input_dim=128, input_length=1, hidden_dim=128, output_length=max_caption_len,
                             output_dim=128, peek=True)
    image_model.add(model)
    image_model.add(TimeDistributed(Dense(vocab_size)))
    image_model.add(Activation('softmax'))
    return image_model