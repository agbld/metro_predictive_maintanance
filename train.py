#%%
from matplotlib import use
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.ops.gen_array_ops import Reshape
import get_dataset
from datetime import date
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

device_keys_table = get_dataset.get_device_keys_table(use_archive=True)
event_keys_table = get_dataset.get_event_keys_table(use_archive=True)

time_steps = 7
time_window_x = 2
time_window_y = 7
X, Y = get_dataset.get_XY_between_date(date(2021, 1, 29), 
                                       date(2021, 4, 19), 
                                       device_keys_table[:50], 
                                       event_keys_table,
                                       time_window_x=time_window_x, 
                                       time_steps=time_steps, 
                                       time_window_y=time_window_y,
                                       use_archive=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

X_train.pop('device_key')
X_train.pop('date')

X_train = np.asarray(X_train).astype('float32')
# X_train = np.array(X_train.reshape((len(X_train), time_steps, 352, 1)))

Y_train = np.asarray(Y_train).astype('float32')
Y_train = np.where(Y_train > 0, 1, 0)

print(Y_train.shape)
print(Y_train.shape)


# %%
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from model_bench import test_model, plot_result

num_of_features = len(X_train[0])

num_of_epochs = 300

model_name = 'model_'

def create_dens_structure():
  model_name = 'model_Den4x_Den4x_1'
  model = Sequential(name=model_name)
  model.add(Dense(num_of_features * 4, activation='relu', input_shape=(num_of_features,)))
  model.add(Dense(num_of_features * 4, activation='relu', input_shape=(num_of_features,)))
  model.add(Dense(1, activation='sigmoid'))
  opt = Adam(lr=1e-4, decay=1e-4)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  # mean_absolute_error
  # binary_crossentropy
  print(model.summary())
  return model

def create_conv_structure():
  model_name = 'model_Con5_Den300_Drp20_1'
  model = Sequential(name=model_name)
  model.add(Reshape((time_steps, 352, 1), input_shape=(len(X_train[0]), )))
  model.add(Conv2D(filters=5, kernel_size=(time_steps, 1), activation='relu', padding='valid'))
  # model.add(Conv2D(filters=256, kernel_size=(1, 352), padding='valid', activation='relu'))
  model.add(Flatten())
  model.add(Dense(300, activation='relu'))  #need to optimize for training efficiancy 
  # model.add(Dropout(0.2))                 #need to optimize for no over fit
  model.add(Dense(1, activation='sigmoid')) #try softmax
  opt = Adam(lr=1e-4, decay=1e-4/2)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  # mean_absolute_error
  # binary_crossentropy
  print(model.summary())
  return model

model = create_conv_structure()

#%%
history = model.fit(X_train, Y_train, epochs=num_of_epochs, validation_split=0.2, shuffle=True)
print('saving model ...')
model.save('bench_model_backup/' + model_name)
print('model saved')

loss_list = history.history['loss']
val_loss_list = history.history['val_loss']
accuracy_list = history.history['accuracy']
val_accuracy_list = history.history['val_accuracy']

# plot_result([loss_list, val_loss_list, accuracy_list, val_accuracy_list], model_name)

import csv
with open('for_remote_training.csv', 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
      for i in range(len(loss_list)):
            writer.writerow([loss_list[i], val_loss_list[i], accuracy_list[i], val_accuracy_list[i]])

#%%
# test_model('model_backup/' + model_name, X_test, Y_test, (len(X_test), time_steps, 352, 1))

#%%