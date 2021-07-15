#%%
from matplotlib import use
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.ops.gen_array_ops import Reshape
import data_transform
from datetime import date
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

device_keys_table = data_transform.get_device_keys_table(use_archive=True, svce_loc_id_list=data_transform.get_station_list_CU307())
event_keys_table = data_transform.get_event_keys_table(use_archive=True)

time_window_x = 2
time_steps = 7
time_window_y = 7
X_train_df, Y_train_df = data_transform.get_XY_between_date(date(2021, 1, 15), 
                                       date(2021, 4, 12), 
                                       device_keys_table, 
                                       event_keys_table,
                                       time_window_x=time_window_x, 
                                       time_steps=time_steps, 
                                       time_window_y=time_window_y,
                                       use_archive=True)
X_test_df, Y_test_df = data_transform.get_XY_between_date(date(2021, 4, 12), 
                                       date(2021, 4, 19), 
                                       device_keys_table, 
                                       event_keys_table,
                                       time_window_x=time_window_x, 
                                       time_steps=time_steps, 
                                       time_window_y=time_window_y,
                                       use_archive=True)
print('X_train_df.shape = ' + str(X_train_df.shape) + '\tY_train_df.shape = ' + str(Y_train_df.shape))
print('X_test_df.shape = ' + str(X_test_df.shape) + '\tY_test_df.shape = ' + str(Y_test_df.shape))

#%%

X_train_df, Y_train_df = data_transform.data_argument(X_train_df, Y_train_df, multiply_y=3)
print('Argumented X_train_df.shape = ' + str(X_train_df.shape) + 'Argumented Y_train_df.shape = ' + str(Y_train_df.shape))

X_train, Y_train = data_transform.convert_to_model_input(X_train_df, Y_train_df)
X_test, Y_test = data_transform.convert_to_model_input(X_test_df, Y_test_df)

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape, Input, concatenate
from tensorflow.keras.optimizers import Adam
from model_bench import test_model, plot_result

num_of_features = len(X_train[0])

num_of_epochs = 200

model_name = 'model_'

def create_simple_dens_structure():
  model_name = 'model_1'
  model = Sequential(name=model_name)
  # model.add(Dense(200, activation='relu', input_shape=(num_of_features,)))
  # model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid', input_shape=(num_of_features,)))
  opt = Adam(lr=1e-4/2, decay=1e-4/2)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  # mean_absolute_error
  # binary_crossentropy
  print(model.summary())
  return model

def create_dens_structure():
  model_name = 'model_Den_Den_Drp_1'
  model = Sequential(name=model_name)
  model.add(Dense(400, activation='relu', 
                  input_shape=(num_of_features,)))
  model.add(Dense(400, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))
  opt = Adam(lr=1e-4, decay=1e-4/2)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  # mean_absolute_error
  # binary_crossentropy
  print(model.summary())
  return model

def create_conv_structure():
  model_name = 'model_Con_Den3_Drp_1'
  model = Sequential(name=model_name)
  model.add(Reshape((time_steps, 353, 1), input_shape=(len(X_train[0]), )))
  model.add(Conv2D(filters=5, kernel_size=(time_steps, 1), activation='relu', padding='valid'))
  # model.add(Conv2D(filters=256, kernel_size=(1, 352), padding='valid', activation='relu'))
  model.add(Flatten())
  model.add(Dense(300, activation='relu'))  #need to optimize for training efficiancy 
  # model.add(Dropout(0.4))                 #need to optimize for no over fit
  model.add(Dense(1, activation='sigmoid')) #try softmax
  opt = Adam(lr=1e-5, decay=1e-4/2)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  # mean_absolute_error
  # binary_crossentropy
  print(model.summary())
  return model

def create_pconv_structure():
  model_name = 'model_parallel'
  
  time_conv_input = Input(shape=(len(X_train[0]), ))
  time_conv = Reshape((time_steps, 353, 1))(time_conv_input)
  time_conv = Conv2D(filters=5, kernel_size=(time_steps, 1), activation='relu', padding='valid')(time_conv)
  time_conv = Flatten()(time_conv)
  time_conv = Dense(250, activation='relu')(time_conv)

  relation_conv_input = Input(shape=(len(X_train[0]), ))
  relation_conv = Reshape((time_steps, 353, 1))(relation_conv_input)
  relation_conv = Conv2D(filters=128, kernel_size=(1, 353), activation='relu', padding='valid')(relation_conv)
  relation_conv = Flatten()(relation_conv)
  relation_conv = Dense(128, activation='relu')(relation_conv)

  concatenate_layer = concatenate([time_conv, relation_conv])
  model_pconv = Dropout(0.4)(concatenate_layer)
  model_pconv = Dense(1, activation='sigmoid')(model_pconv)
  
  model_pconv = Model([time_conv_input, relation_conv_input], model_pconv)

  opt = Adam(lr=1e-4, decay=1e-3)
  model_pconv.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  # mean_absolute_error
  # binary_crossentropy
  print(model_pconv.summary())
  return model_pconv

model = create_dens_structure()

#%%
history = model.fit(X_train, Y_train, epochs=num_of_epochs, batch_size=400, validation_data=(X_test, Y_test), shuffle=True)
model.evaluate(X_test, Y_test)
print('saving model ...')
model.save('bench_model_backup/' + model_name)
print('model saved')

loss_list = history.history['loss']
val_loss_list = history.history['val_loss']
accuracy_list = history.history['accuracy']
val_accuracy_list = history.history['val_accuracy']

plot_result([loss_list, val_loss_list, accuracy_list, val_accuracy_list], model_name)

# import csv
# with open('for_remote_training.csv', 'w', newline='') as csvfile:
#       writer = csv.writer(csvfile)
#       writer.writerow(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
#       for i in range(len(loss_list)):
#             writer.writerow([loss_list[i], val_loss_list[i], accuracy_list[i], val_accuracy_list[i]])

#%%

sample_X = X_test
sample_Y = Y_test

predictions = model.predict(sample_X)
correctness = 0
pred_0 = 0
pred_1 = 0
correctness_pred_0 = 0
correctness_pred_1 = 0

for i in range(len(sample_X)):
  pred_raw = round(predictions[i][0], 2)
  pred = 0
  if pred_raw > 0.5 : pred = 1
  y = sample_Y[i][0]
  diff = abs(y - pred)
  correctness += 1 - diff
  pred_0 += 1 - pred
  pred_1 += pred
  if pred == 0 and pred == y: correctness_pred_0 += 1
  if pred == 1 and pred == y: correctness_pred_1 += 1
  row = [pred_raw, pred, y, diff]
  if diff == diff:
    print(str(i) + 'th' + '\t' + '\tpred_raw:' + str(pred_raw) + '\tpred:' + str(pred) + '\tY:' + str(y) + '\tdiff:' + str(diff))

correctness = round(correctness / (pred_0 + pred_1), 4) * 100
correctness_pred_0 = round(correctness_pred_0 / pred_0, 4) * 100
correctness_pred_1 = round(correctness_pred_1 / pred_1, 4) * 100
print('correctness: ' + str(correctness) + '%\tcorrectness(pred:0): ' + str(correctness_pred_0) + '%\tcorrectness(pred:1): ' + str(correctness_pred_1) + '%')

#%%