# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018/11/17/
    项目名称： 猫狗大战
    版本： V 0.0
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import seaborn as sns
import os,cv2,random
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,Dropout,Activation
import matplotlib as mpl
from keras import optimizers
import tensorflow
mpl.rcParams['font.sans-serif'] = ['SimHei']
train_path = './train/'
test_path = './test/'
ROWS = 256
COLS = 256
DIM = 3
train_images = [train_path + i for i in os.listdir(train_path)]
test_images = [test_path + i for i in os.listdir(test_path)]
train_cat = [train_path + i for i in os.listdir(train_path) if 'cat' in i]
train_dog = [train_path + i for i in os.listdir(train_path) if 'dog' in i]
count_train = len(train_images)
labels = []
train_set = np.ndarray((count_train,ROWS,COLS,DIM),dtype=np.uint8)
# train_set = np.ndarray((1000,ROWS,COLS,DIM),dtype=np.uint8)
for i,img_file in enumerate(train_images):
    img = cv2.imread(img_file)
    img = cv2.resize(img,(ROWS,COLS),interpolation=cv2.INTER_CUBIC)
    train_set[i] = img
    if 'cat' in img_file:
        labels.append(0)
    else:
        labels.append(1)
test_set = np.ndarray((len(test_images),ROWS,COLS,DIM),dtype=np.uint8)
# test_set = np.ndarray((1000,ROWS,COLS,DIM),dtype=np.uint8)
for i,img_file in enumerate(test_images):
    img = cv2.imread(img_file)
    img = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    test_set[i] = img
# print(train_set.shape)
# print("-"*32)
# print(len(labels))
def creat_model():
    model = keras.Sequential()
    model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(ROWS,COLS,3),activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(64,3,3,border_mode='same',activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    optimizers_cd = optimizers.SGD(lr=0.0001)
    model.compile(optimizer=optimizers_cd,loss='binary_crossentropy')
    return model
model = creat_model()
epochs= 20
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []


    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
def run_model():

    history = LossHistory()
    model.fit(x=train_set,y=labels,batch_size=16,epochs=epochs,validation_split=0.25,verbose=0,shuffle=True,callbacks=[history])
    prediction = model.predict(test_set,verbose=0)
    return history,prediction
history,prediction = run_model()
print(history.losses)
# plt.plot(epochs_label,history.losses,'b',labels='Training Loss')
# plt.plot(epochs_label,history.val_losses,'r',labels='Validation Loss')
# plt.xlabel('epochs')
# plt.ylabel('error')
# plt.show()
prediction = pd.DataFrame(prediction)
prediction.to_csv('submission.csv')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(history.losses, 'blue', label='Training Loss')
plt.plot(history.val_losses, 'green', label='Validation Loss')
plt.xticks(range(0,epochs)[0::2])
plt.legend()
plt.show()



