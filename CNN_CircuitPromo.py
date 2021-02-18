import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Union

import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from sklearn.model_selection import KFold, StratifiedKFold
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os
import glob


def get_cnn1d_model(nb_feature: int, nb_class: int,
                    metric: str    = 'accuracy',
                    optimizer: str = 'adam',
                    loss: str      = 'categorical_crossentropy'):
    # create the model                
    model = Sequential()
    model.add(Convolution1D(filters=64, kernel_size=3, input_shape=(nb_feature, 1)))
    model.add(Activation('relu'))
    model.add(Convolution1D(filters=64, kernel_size=3))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

res_use = []
users = glob.glob("groups/gr1/*")
for user in users:
  file_path = glob.escape(user +'/')

  train_df = pd.read_csv(f'{file_path}/zsynth_train.csv', index_col=0)\
    .drop(['user_id', 'order_id', 'product_id'], axis=1)\
    .reset_index(drop=True)
  cv_scores  = []
  cv_prec = []
  cv_rec = []
  features   =  [feature for feature in train_df.columns if feature != 'c']
  nb_class   = len(train_df.c.unique())
  model_type = 'cnn'
  kfold = StratifiedKFold(n_splits=10, shuffle=True)
  train_y = train_df.c
  X = train_df[features]
  y = train_y
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  # cv run
  for train_idx, test_idx in kfold.split(np.zeros(len(X_train)), y_train):
      _df_copy        = train_df.copy()
      training_data   = _df_copy.iloc[train_idx]
      validation_data = _df_copy.iloc[test_idx]

      _x_training  = training_data[features]
      _y_training  = training_data[['c']]

      _x_validation = validation_data[features]
      _y_validation = validation_data[['c']]

      if model_type == 'cnn':
          _x_training_shaped   = _x_training.to_numpy().reshape((len(_x_training), len(features), 1))
          _x_validation_shaped = _x_validation.to_numpy().reshape((len(_x_validation), len(features), 1))

          _y_training_cat   = np_utils.to_categorical(_y_training, nb_class)
          _y_validation_cat = np_utils.to_categorical(_y_validation, nb_class)

          model = get_cnn1d_model(len(features), nb_class)
          
          model.fit(_x_training_shaped, 
              _y_training_cat, epochs=100, 
              batch_size=64, verbose=0, 
              validation_data=(_x_validation_shaped, _y_validation_cat))

          scores = model.evaluate(_x_validation_shaped, _y_validation_cat, verbose=0)
           # display and add model perf
      score_percentage = scores[1] * 100
      prec = scores[2] * 100 
      rec = scores [3] * 100
      print("%s: %.4f%%" % (model.metrics_names[1], score_percentage))
      cv_scores.append(score_percentage)
      cv_prec.append(prec)
      cv_rec.append(rec)


print("mean Accuracy: %.3f%%, stdv: %.3f%%" % (np.mean(cv_scores), np.std(cv_scores)))
print("mean Precision: %.3f%%, stdv: %.3f%%" % (np.mean(cv_prec), np.std(cv_prec)))
print("mean Recall: %.3f%%, stdv: %.3f%%" % (np.mean(cv_rec), np.std(cv_rec)))
mean_f = (2*np.mean(cv_rec)*np.mean(cv_prec))/(np.mean(cv_prec)+np.mean(cv_rec))
print("Mean F score : %.3f%%" %mean_f)



results = model.evaluate(_x_test_shaped, y_test_cat, batch_size= 64, verbose=0)
  

print("test results  ")
for name, value in zip(model.metrics_names, results):
	print(name, ': ', value)
    if name == 'accuracy' :
      	res_use.append(value)
    	print()

print(classification_report(y))

results = model.evaluate(_x_test_shaped, y_test_cat, batch_size= 32, verbose=0)

print("test results  ")
for name, value in zip(model.metrics_names, results):
  print(name, ': ', value)
print()

classification_report(y)
test_pred = model.predict(_x_test_shaped, batch_size=32)
a = test_pred > 0.5

print(classification_report(y_test, aa))
