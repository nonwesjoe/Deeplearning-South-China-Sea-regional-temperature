import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Dropout, Activation, multiply, Flatten, Attention, Conv2D, ConvLSTM2D, LSTM, \
    MaxPooling2D, BatchNormalization, Input, Multiply, Lambda, TimeDistributed, Conv3D
# tensorflow 2.17.0, keras 3.4.1
with h5py.File('/kaggle/input/southseadata/southsea.h5','r') as f:
    X=f['data'][:]

print(len(X))
# 设置时间步长
win = 3
x, y = [], []
for i in range(win, len(X)-win):
    x.append(X[i-win:i])
    y.append(X[i])
X=[]
x, y = np.array(x), np.array(y)

x_train=x[0:105120]
y_train=y[0:105120]
x_test=x[105120:]
y_test=y[105120:]

#释放内存
x=[]
y=[]


input_shape = (3, 69, 53, 1)

model = Sequential()
model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                     input_shape=input_shape,
                     padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                     padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                     padding='same', return_sequences=False)) 
model.add(BatchNormalization())

model.add(Conv2D(filters=1, kernel_size=(3, 3),
                 activation='linear',
                 padding='same', data_format='channels_last'))
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

checkpoint =ModelCheckpoint(filepath='era5.keras',  # 保存的模型文件名
                                 monitor='val_loss',  # 监控的指标，可以选择其他指标如'val_accuracy'
                                 verbose=1,  # 显示保存信息
                                 save_best_only=True,  # 仅保存性能最好的模型
                                 mode='min')

history = model.fit(x_train, y_train, epochs=40, batch_size=64, validation_data=(x_test, y_test),callbacks=[checkpoint])
