#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape, Input, concatenate
from tensorflow.keras.optimizers import Adam
def create_dens_structure(num_of_features):
  model_name = 'model_' + str(datetime.today())
  model = Sequential(name=model_name)
  model.add(Dense(400, activation='relu', input_shape=(num_of_features,)))
  model.add(Dense(400, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))
  opt = Adam(lr=1e-4, decay=1e-4/2)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  print(model.summary())
  return model

#%%
import data_transform
from datetime import date, datetime, timedelta
from pandas import DataFrame
from matplotlib import pyplot as plt
def train_model_for_date(date:date, device_type='', use_archive=True, plot_result=True):
  print('training model for ' + str(date) + ' prediction ...')
  
  # get device_key_table and event_key_table
  if device_type == 'CU307':
    device_keys_table = data_transform.get_device_keys_table(use_archive=True, svce_loc_id_list=data_transform.get_station_list_CU307())
  elif device_type == 'SongXin':
    device_keys_table = data_transform.get_device_keys_table(use_archive=True, svce_loc_id_list=data_transform.get_station_list_SongXin())
  elif device_type == 'Circular':
    device_keys_table = data_transform.get_device_keys_table(use_archive=True, svce_loc_id_list=data_transform.get_station_list_Circular())
  else:
    device_keys_table = data_transform.get_device_keys_table(use_archive=True)
  event_keys_table = data_transform.get_event_keys_table(use_archive=True)
  
  # collect all needed data from history
  X_train_df, Y_train_df = data_transform.get_practical_XY_train_of_date(date, device_keys_table, event_keys_table, trace_back_to_week=6)
  
  # show data statistics
  train_rows = Y_train_df['target'].count()
  train_y1_perc = round(Y_train_df[Y_train_df['target'] == 1].count() / train_rows * 100, 2)
  print('for training data, {0}% row has y == 1'.format(train_y1_perc))
  
  # data argumentation and convert to model input
  X_train_df, Y_train_df = data_transform.data_argument(X_train_df, Y_train_df, multiply_y=3)
  X_train, Y_train = data_transform.convert_to_model_input(X_train_df, Y_train_df)
  
  # create and train model
  model = create_dens_structure(num_of_features=len(X_train[0]))
  history = model.fit(X_train, Y_train, epochs=200, batch_size=int(train_rows / 200), validation_split=0.2, shuffle=True)

  # plot result
  if plot_result:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('date')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('date')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
  
  return model

#%%
def make_prediction_with_model(model_name:str):
  pass