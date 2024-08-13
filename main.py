import os
import random
import keras
import tensorflow as tf


tf.compat.v1.disable_eager_execution()
import scipy.io as sio
import numpy as np
import pandas as pd
from keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Activation, Conv1D, MaxPooling1D,AveragePooling1D, Dropout, Lambda, LeakyReLU, \
    LSTM, Input
from tensorflow.python.keras.optimizer_v1 import SGD, Adam, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import sys


print(sys.version)


# ## R^2 coefficient function

# In[ ]:


# --------------------------------R^2 coefficient for variance------------
# - the higher the R-squared, the better the model fits your data.
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))





# ##Callbacks

# In[ ]:


# -----------------------------------------callbacks---------------------------
# 1. Epoch Schedule
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_mean_absolute_error') < 3) and (logs.get('mean_absolute_error') < 3) and (
                logs.get('val_loss') < 18) and (logs.get('loss') < 18):
            print("\nReached perfect accuracy so cancelling training!")
            self.model.stop_training = True


epoch_schedule = myCallback()

# 2. Learning Rate Schedule
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-6 * 10 ** (epoch / 2.5))


# In[ ]:


def _normalize(data, mode='max'):

    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert (max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data

    elif mode == 'std':
        std_data = np.std(data, axis=0, keepdims=True)
        assert (std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data


# ## Importing data

# In[ ]:


df=pd.read_excel('G:\Article3\data\Absorption.xlsx', header=None)   # import dataset

print(df.shape[0])

# ## Deciding Features and Labels

# In[ ]:


# Input/Features and labels extraction
X1 = df.iloc[:, 4:]
print(X1)
df_S=df.iloc[:, 0:4]
X = np.array(X1)

# In[ ]:


X = _normalize(X, mode='max')
print(X.shape)

# In[ ]:


y = df_S.iloc[:, 0:df_S.shape[1]]
print(y)
y = np.array(y)


# ##Splitting testing and test set

# In[ ]:



# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=83, shuffle=True)
X_valid = X_test
y_valid = y_test

# In[ ]:


m_train = X_train.shape[0]
m_valid = X_valid.shape[0]
n_steps = X.shape[1]
n_features = 1  # or 2
n_outputs = y_train.shape[1]
print(m_train)
print(m_valid)
print(n_steps)
print(n_features)
print(n_outputs)

# ##Reshape the arrays

# In[ ]:


# reshape the arrays
X_train = np.reshape(X_train, (m_train, n_steps, n_features))
X_valid = np.reshape(X_valid, (m_valid, n_steps, n_features))

# ## Model Architecture

# In[ ]:


# define model architecture : 1DCNN
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=7, input_shape=(n_steps, n_features),
                 kernel_regularizer=keras.regularizers.l2(1e-6), bias_regularizer=keras.regularizers.l1(1e-4)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(pool_size=2,strides=1))
model.add(Conv1D(filters=32, kernel_size=5, kernel_regularizer=keras.regularizers.l2(1e-6),
                 bias_regularizer=keras.regularizers.l1(1e-4)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(pool_size=2,strides=1))
model.add(Conv1D(filters=64, kernel_size=3,kernel_regularizer=keras.regularizers.l2(1e-6),
                 bias_regularizer=keras.regularizers.l1(1e-4)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(pool_size=2,strides=1))
model.add(Flatten())
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.1))
model.add(Dense(4, activation='linear'))

# ## Choose Hyperparameters

# In[ ]:


# compile model
model.compile(loss='mse', optimizer=Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-5), metrics=['mae', r_square])

# ## Train the model

# In[ ]:


batchsize = 35
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batchsize, epochs=200, verbose=1, shuffle=True)

# In[ ]:





# ## Plot Accuracy and Loss

# ###MSE

# In[ ]:


# -----------------------------------------------Summarize history for loss--------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], '-o')
plt.plot(history.history['val_loss'], '-s')
plt.title('Loss curve for 1D-CNN', fontsize=18)
plt.ylabel('MSE loss', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Number of epochs', fontsize=18)
plt.legend(['train', 'test'], loc='upper right', fontsize=18)
plt.show()

MSE_train=pd.DataFrame(history.history['loss'])
MSE_train.to_csv('MSE_train.csv',index=False)
MSE_test=pd.DataFrame(history.history['val_loss'])
MSE_test.to_csv('MSE_test.csv',index=False)
# ###MAE

# In[ ]:


# ---------------------------------------Summarize history for MAE------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'], '-o')
plt.plot(history.history['val_mae'], '-s')
plt.title('model MAE', fontsize=18)
plt.ylabel('MAE', fontsize=18)
plt.xlabel('epoch', fontsize=18)
plt.legend(['train', 'test'], loc='upper left', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

MAE_train=pd.DataFrame(history.history['mae'])
MAE_train.to_csv('MAE_train.csv',index=False)
MAE_test=pd.DataFrame(history.history['val_mae'])
MAE_test.to_csv('MAE_test.csv',index=False)

# ###R^2 coefficient

# In[ ]:


# ----------------------------------------Summarize history for R^2------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history['r_square'], '-o')
plt.plot(history.history['val_r_square'], '-s')
plt.title('model R^2', fontsize=18)
plt.ylabel('R^2', fontsize=18)
plt.xlabel('epoch', fontsize=18)
plt.legend(['train', 'test'], loc='upper left', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

R_train=pd.DataFrame(history.history['r_square'])
R_train.to_csv('R_train.csv',index=False)
R_test=pd.DataFrame(history.history['val_r_square'])
R_test.to_csv('R_test.csv',index=False)
# ## Get learned weights and biases

# ###Weights

# In[ ]:

print('this is layer_weights')
layer_weights = model.layers[0].get_weights()[0]
print(layer_weights.shape)
print(layer_weights)

# ###Biases

# In[ ]:

print('this is layer_biases')
layer_biases = model.layers[0].get_weights()[1]
print(layer_biases.shape)
print(layer_biases)

# In[ ]:


# ------------------------End of the code------------------------#


pre_A0 = pd.read_excel('G:\Article3\data\Pre.xlsx', header=None)

preX = np.array(pre_A0)
preY = _normalize(preX, mode='max')
preZ = np.reshape(preY, (1, 71, 1))
preC = model.predict(preZ)
print('this is structure parameters')
print(preC)

model.summary()