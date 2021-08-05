#%%
from numpy import mod
import sqlalchemy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import device
from tensorflow.python.keras.engine.training import write_scalar_summaries
def create_dens_structure(name:str, num_of_features:int):
  model = Sequential(name=name)
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
def train_model_for_date(date:date, device_type='', epochs=200, use_archive=False, plot_result=True):
  print('training model for ' + str(date) + ' prediction ...')
  
  # get device_key_table and event_key_table
  device_keys_table = []
  if device_type == 'CU307':
    device_keys_table = data_transform.get_device_keys_table(use_archive=use_archive, svce_loc_id_list=data_transform.get_station_list_CU307())
  elif device_type == 'SongXin':
    device_keys_table = data_transform.get_device_keys_table(use_archive=use_archive, svce_loc_id_list=data_transform.get_station_list_SongXin())
  elif device_type == 'Circular':
    device_keys_table = data_transform.get_device_keys_table(use_archive=use_archive, svce_loc_id_list=data_transform.get_station_list_Circular())
  else:
    device_keys_table = data_transform.get_device_keys_table(use_archive=use_archive)
  event_keys_table = data_transform.get_event_keys_table(use_archive=use_archive)
  
  # collect all needed data from history
  X_train_df, Y_train_df = data_transform.get_practical_XY_train_of_date(date, device_keys_table, event_keys_table, trace_back_to_week=6, use_archive=use_archive)
  
  # show data statistics
  train_rows = Y_train_df['target'].count()
  train_y1_perc = int(Y_train_df[Y_train_df['target'] == 1].count() / train_rows * 100)
  print('for original data, {0}% row has y == 1\n'.format(str(train_y1_perc)))
  
  # data argumentation and convert to model input
  X_train_df, Y_train_df = data_transform.data_argument(X_train_df, Y_train_df, multiply_y=3)
  train_rows = Y_train_df['target'].count()
  train_y1_perc = int(Y_train_df[Y_train_df['target'] == 1].count() / train_rows * 100)
  print('for argumented data, {0}% row has y == 1\n'.format(str(train_y1_perc)))
  
  X_train, Y_train = data_transform.convert_to_model_input(X_train_df, Y_train_df)
  
  # create and train model
  model = create_dens_structure('model_' + str(date), num_of_features=len(X_train[0]))
  history = model.fit(X_train, Y_train, epochs=epochs, batch_size=int(train_rows / 200), validation_split=0.2, shuffle=True)
  model.save('trained_model/' + model.name)

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
def update_archive():
  device_keys_table = data_transform.get_device_keys_table(use_archive=False)
  event_keys_table = data_transform.get_event_keys_table(use_archive=False)
  date_oldest = date(2021, 2, 1)
  date_latest = date(2021, 4, 19)
  X, Y = data_transform.get_XY_between_date(date_oldest, date_latest, device_keys_table, event_keys_table, use_archive=False)

#%%
from tensorflow.keras.models import load_model
import csv
import pymssql

mssql_host=''
mssql_user=''
mssql_password=''
mssql_database=''

with open('refer/mssql_info.csv', newline='') as csvfile:
  rows = csv.reader(csvfile)
  mssql_host=next(rows)[0]
  mssql_user=next(rows)[0]
  mssql_password=next(rows)[0]
  mssql_database=next(rows)[0]

def make_prediction_of_the_date_with_model(make_prediction_date:date, model_path:str, result_path='result.csv', device_type='', append=True, use_archive=False):
  
  # get device_key_table and event_key_table
  device_keys_table = []
  if device_type == 'CU307':
    device_keys_table = data_transform.get_device_keys_table(use_archive=use_archive, svce_loc_id_list=data_transform.get_station_list_CU307())
  elif device_type == 'SongXin':
    device_keys_table = data_transform.get_device_keys_table(use_archive=use_archive, svce_loc_id_list=data_transform.get_station_list_SongXin())
  elif device_type == 'Circular':
    device_keys_table = data_transform.get_device_keys_table(use_archive=use_archive, svce_loc_id_list=data_transform.get_station_list_Circular())
  else:
    device_keys_table = data_transform.get_device_keys_table(use_archive=use_archive)
  event_keys_table = data_transform.get_event_keys_table(use_archive=use_archive)
  
  # get needed featurs and transform
  X_df, Y_df = data_transform.get_XY_between_date(make_prediction_date, make_prediction_date + timedelta(days=1), device_keys_table, event_keys_table, use_archive=use_archive)
  # X_df = DataFrame(data_transform.get_X_of_the_date(make_prediction_date, device_keys_table, event_keys_table))
  X_input, Y_input = data_transform.convert_to_model_input(X_df, Y_df)
  
  model = load_model(model_path)
  predictions = model.predict(X_input)
  
  predict_from_date = make_prediction_date + timedelta(days=1)
  predict_to_date = make_prediction_date + timedelta(days=8)
  result = []
  write_mode = 'a'
  if not append: write_mode = 'w'
  
  # conn = pymssql.connect(
  #   host=mssql_host,
  #   user=mssql_user,
  #   password=mssql_password,
  #   database=mssql_database,
  # )
  # cursor = conn.cursor(as_dict=True)
  cmd = 'INSERT INTO prediction (device_key, svce_loc_id, dev_name, from_date, to_date, prediction) VALUES '
  
  with open(result_path, write_mode, newline='') as csvfile:
    writer = csv.writer(csvfile)
    # writer.writerow(['device_key', 'svce_loc_id', 'dev_name', 'from_date', 'to_date', 'predict'])
    for i in range(len(X_df)):
      device_key = X_df.iloc[i]['device_key']
      prediction = round(predictions[i][0], 2)
      result_tmp = {'device_key':device_key, 'svce_loc_id': device_key[:3], 'dev_name': device_key[3:], 'from_date':str(predict_from_date), 'to_date':str(predict_to_date), 'predict':str(prediction)}
      # print('device {0} has prob. {1} malfunction bewteen {2} and {3}.'.format(device_key, str(prediction), str(predict_from_date), str(predict_to_date)))
      result.append(result_tmp)
      writer.writerow([result_tmp['device_key'], result_tmp['svce_loc_id'], result_tmp['dev_name'], result_tmp['from_date'], result_tmp['to_date'], result_tmp['predict']])
      cmd += "('{0}', '{1}', '{2}', '{3}', '{4}', {5}),".format(result_tmp['device_key'], 
                                                              result_tmp['svce_loc_id'], 
                                                              result_tmp['dev_name'],
                                                              result_tmp['from_date'],
                                                              result_tmp['to_date'],
                                                              result_tmp['predict'])
  cmd = str(cmd[:-1])
  # print(cmd)
  # cursor.execute(cmd)
  # conn.commit()
  # cursor.close()
  # conn.close()
  result_df = DataFrame(result)
  conn = sqlalchemy.engine.URL.create(
      "mssql+pymssql",
      username=mssql_user,
      password=mssql_password,
      host=mssql_host,
      database=mssql_database,
  )
  conn = sqlalchemy.create_engine(conn, echo=False)
  result_df.to_sql(con=conn, name="prediction", if_exists='append', index=False, method='multi')
  return DataFrame(result)

#%%