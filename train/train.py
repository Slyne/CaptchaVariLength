#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.callbacks import ModelCheckpoint, Callback
import numpy as np
from utils import load_data, create_simpleCnnRnn

class ValidateAcc(Callback):
    def __init__(self, image_model, val_data, val_label):
        self.image_model = image_model
        self.val = val_data
        self.val_label = val_label

    def on_epoch_end(self, epoch, logs={}):
        print '\n———————————--------'
        self.image_model.load_weights('../checkpoints/weights.%02d.hdf5' % epoch)
        r = self.image_model.predict(val, verbose=0)
        y_predict = np.asarray([np.argmax(i, axis=1) for i in r])
        val_true = np.asarray([np.argmax(i, axis = 1) for i in self.val_label])
        length = len(y_predict) * 1.0
        correct = 0
        for (true,predict) in zip(val_true,y_predict):
            print true,predict
            if list(true) == list(predict):
                correct += 1
        print "Validation set acc is: ", correct/length
        print '\n———————————--------'

from config import image_shape,max_caption_len,vocab_size
image_model = create_simpleCnnRnn(image_shape, max_caption_len,vocab_size)
split_ratio = 0.7
train,train_label,val,val_label = load_data(split_ratio)

val_acc_check_pointer = ValidateAcc(image_model,val,val_label)
check_pointer = ModelCheckpoint(filepath="../checkpoints/weights.{epoch:02d}.hdf5")

image_model.fit(train, train_label,
                shuffle=True, batch_size=16, nb_epoch=20, callbacks=[check_pointer, val_acc_check_pointer])

#image_model.save("../checkpoints/model2.hdf5")
