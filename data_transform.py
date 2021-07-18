#%%
from os import path
import datetime
from datetime import date, timedelta
import csv
from pandas.core.frame import DataFrame
import pymssql
from scipy.sparse import data
from tensorflow.python.framework.ops import device

event_key_table_path = 'refer/事件警報對照表_toAFC.csv'
collected_data_folder = 'collected_data/'
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

#%%
def get_station_list_CU307():
  station_list = ['7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '45', '46', '47', '48', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '174', '175', '176', '177', '178', '179', '180']
  for i in range(len(station_list)):
    station_list[i] = str(station_list[i]).zfill(3)
  return station_list

def get_station_list_SongXin():  
  station_list = ['76', '99', '100', '101', '102', '103', '105', '106', '107', '108', '109', '110', '111']
  for i in range(len(station_list)):
    station_list[i] = str(station_list[i]).zfill(3)
  return station_list

def get_station_list_Circular():
  station_list = ['200', '201', '202', '203', '205', '206', '207', '208', '209', '210', '211', '212', '213']
  for i in range(len(station_list)):
    station_list[i] = str(station_list[i]).zfill(3)
  return station_list

#%%
# collect all device_key(s) from server, and return a distincted device_keys_table.
# (optional) given a not empty [svce_loc_id_list] to collect devices in those stations list in [svce_loc_id_list]
# (optional) set [use_archive]=False to always collect data from server insteed of csv
# device_key : [svce_loc_id] + [dev_name]
def get_device_keys_table(host=mssql_host, user=mssql_user, passward=mssql_password, database=mssql_database, svce_loc_id_list=[], use_archive=True):
  print('\ncollecting device key table ...')
  key_table = [] # the device key table
  save_path = collected_data_folder + 'device_keys_table.csv'
  if path.exists(save_path) and use_archive:
    with open(save_path, newline='') as csvfile:
      rows = csv.reader(csvfile)
      for row in rows:
        if len(svce_loc_id_list) == 0:
          key_table.append(row[0])
        elif str(row[0][0:3]) in svce_loc_id_list:
          key_table.append(row[0])
    print('device key table loaded from csv archive.')
    return key_table
  
  print('csv archive not found, collecting from server ...')
  conn = pymssql.connect(
    host=host,
    user=user,
    password=passward,
    database=database,
  )
  cursor = conn.cursor(as_dict=True)
  
  
  if len(svce_loc_id_list) > 0:
    print('collecting devices with specific stations list ...')
  
  # append table with event log
  cmd_from_log = "SELECT svce_loc_id, dev_name FROM [ATIM_event_alarm] where dev_name not like '%VATIM%' group by svce_loc_id, dev_name"
  cursor.execute(cmd_from_log)
  for row in cursor:
    svce_loc_id = str(row['svce_loc_id']).zfill(3)
    if len(svce_loc_id_list) == 0:
      key_table.append(svce_loc_id + row['dev_name'])
    elif svce_loc_id in svce_loc_id_list:
      key_table.append(svce_loc_id + row['dev_name'])
  
  # append table with view
  # cmd_from_view = "SELECT svce_loc_id, dev_name FROM ATIM工單_view where dev_name not like '%VATIM%' group by svce_loc_id, dev_name"
  # cursor.execute(cmd_from_view)
  # for row in cursor:
  #   svce_loc_id = str(row['svce_loc_id']).zfill(3)
  #   if len(svce_loc_id_list) == 0:
  #     key_table.append(svce_loc_id + row['dev_name'])
  #   elif svce_loc_id in svce_loc_id_list:
  #     key_table.append(svce_loc_id + row['dev_name'])
  
  cursor.close()
  conn.close()
  
  # distinct table
  key_table = list(set(key_table))
  print('device key table loaded from server.')
  
  with open(save_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in key_table:
      writer.writerow([row])
  print('device key table archived.')
  
  return key_table

# device_keys_table = get_device_keys_table(svce_loc_id_list=['86',])

#%%
# collect all event_key(s) from server and referance [csv_path], and return a distincted event_keys_table.
# (optional) set [use_archive]=False to always collect data from server insteed of csv
# event_key : [tag_name] + [param_value]
def get_event_keys_table(csv_path=event_key_table_path, host=mssql_host, user=mssql_user, passward=mssql_password, database=mssql_database, use_archive=True):
  print('\ncollecting event key table ...')
  key_table = [] # the event key table
  save_path = collected_data_folder + 'event_keys_table.csv'
  if path.exists(save_path) and use_archive:
    with open(save_path, newline='') as csvfile:
      rows = csv.reader(csvfile)
      for row in rows:
        key_table.append(row[0])
        # print(row)
    print('event key table loaded from csv archive.')
    return key_table
  
  print('csv archive not found, collecting from server ...')
  # append table with csv
  with open(csv_path, newline='', encoding="utf-8") as csvfile:
    rows = csv.reader(csvfile)
    index = 0
    for row in rows:
      if index < 5: 
        index += 1
        continue
      tmp = row[1] + row[2]
      key_table.append(tmp)
  
  # appen table with event log
  conn = pymssql.connect(
    host=host,
    user=user,
    password=passward,
    database=database,
  )
  cursor = conn.cursor(as_dict=True)
  cmd = "select concat(tag_name, param_value) as event_key from ATIM_event_alarm group by concat(tag_name, param_value)"
  cursor.execute(cmd)
  for event_key in cursor:
    key_table.append(event_key['event_key'])
  cursor.close()
  conn.close()
  
  # distinct table
  key_table = list(set(key_table))
  print('event key table loaded from server.')
  
  with open(save_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in key_table:
      writer.writerow([row])
  print('event key table archived.')
  
  return key_table

# event_keys_table = get_event_keys_table(event_key_table_path, mssql_host, mssql_user, mssql_password, mssql_database)

#%%
# in progress ...
def get_malfunction_keys_table(host=mssql_host, user=mssql_user, passward=mssql_password, database=mssql_database):
  print('\ncollecting malfunction key table ...')
  
  key_table = [] # the malfunction keys table
  save_path = collected_data_folder + 'malfuncion_keys_table.csv'
  if path.exists(save_path):
    with open(save_path, newline='') as csvfile:
      rows = csv.reader(csvfile)
      for row in rows:
        key_table.append(row)
    print('malfunction key table loaded from csv archive.')
    return key_table
  
  print('csv archive not found, collecting from server ...')
  conn = pymssql.connect(
    host=host,
    user=user,
    password=passward,
    database=database,
  )
  cursor = conn.cursor(as_dict=True)
  
  # append table with event log
  cmd = "SELECT [故障分類1] FROM [ATIM工單_view] group by [故障分類1] order by [故障分類1]"
  cursor.execute(cmd)
  for row in cursor:
    key_table.append(row['故障分類1'])
  
  cursor.close()
  conn.close()
  
  print('malfunction key table loaded from server.')
  
  with open(save_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in key_table:
      writer.writerow([row])
  print('malfunction key table archived.')
  
  return key_table
# malfunciont_keys_table_tmp = get_malfunction_keys_table()

#%%
# get X of the date:
# collect all needed features for predict y(s) of the specific [date] and specific device(s) in [device_keys_table] according to specific [time_window], [time_steps]
# features:
# assume event_keys_table = ['A', 'B', 'C'], time_steps=2
# features of a device = [[device_key], 
#                         [date],
#                         [count(A) between 1st time window], 
#                         [count(A) between 2st time window], 
#                         [count(B) between 1st time window], 
#                         [count(B) between 2st time window], 
#                         [count(C) between 1st time window], 
#                         [count(C) between 2st time window]]
def get_X_of_the_date(date: date, device_keys_table, event_keys_table, time_window=7, time_steps=4, host=mssql_host, user=mssql_user, passward=mssql_password, database=mssql_database):
  
  conn = pymssql.connect(
    host=host,
    user=user,
    password=passward,
    database=database,
  )
  cursor = conn.cursor(as_dict=True)
  X = {}
  value_counts = 0
  
  for device_key in device_keys_table:
    x = {'device_key':device_key, 'date':str(date)}
    for event_key in event_keys_table:
      for i in range(1, time_steps+1):
        x[event_key + "_" + str(i)] = 0
        value_counts += 1
    for i in range(1, time_steps+1):
      x['malfunc_' + str(i)] = 0
      value_counts += 1
    X[device_key] = x
  
  error_counts = 0
  query_counts = 0
  
  cmd_key_constrain = ''
  if len(device_keys_table) <= 15:
    cmd_key_constrain = ' and (concat(RIGHT(\'000\' + CONVERT(VARCHAR(3),[svce_loc_id]), 3), dev_name) = \'' + device_keys_table[0] + '\''
    for key in device_keys_table[1:]:
      cmd_key_constrain += ' or concat(RIGHT(\'000\' + CONVERT(VARCHAR(3),[svce_loc_id]), 3), dev_name) = \'' + key + '\''
    cmd_key_constrain += ')'
  
  cmd = ""
  for i in range(1, time_steps + 1):
    cmd += """
    select RIGHT('000' + CONVERT(VARCHAR(3),[svce_loc_id]), 3) as svce_loc_id, dev_name, concat(tag_name, param_value) as event_key, \'{0}\' as week, count(concat(tag_name, param_value)) as counts
    from [ATIM_event_alarm]
    where dev_name not like \'%VATIM%\' and create_datetime between \'{1}\' and \'{2}\'{3}
    group by concat(tag_name, param_value), svce_loc_id, dev_name
    """.format(i, str(date - timedelta(days=i * time_window)), str(date + timedelta(days=1) - timedelta(days=(i - 1) * time_window)), cmd_key_constrain)  
    if i < time_steps: cmd += " union "
  
  # print(cmd)
  cursor.execute(cmd)
  
  for row in cursor:
    query_counts += 1
    device_key = str(row['svce_loc_id']) + row['dev_name']
    event_key = row['event_key'] + "_" + row['week']
    if device_key in X:
      X[device_key][event_key] = int(row['counts'])
    else: error_counts += 1
    
  
  cmd = ""
  for i in range(1, time_steps + 1):
    cmd += """
    select concat(svce_loc_id, dev_name) as device_key, \'{0}\' as week, count(dev_name) as counts
    from ATIM工單_view
    where dev_name not like \'%VATIM%\' and 報修日期 between \'{1}\' and \'{2}\'
    group by concat(svce_loc_id, dev_name)
    """.format(i, str(date - timedelta(days=i * time_window)), str(date + timedelta(days=1) - timedelta(days=(i - 1) * time_window)))
    if i < time_steps: cmd += " union "

  cursor.execute(cmd)  
  for row in cursor:
    query_counts += 1
    if row['device_key'] in X:
      X[row['device_key']]['malfunc_' + str(row['week'])] = int(row['counts'])
    else: error_counts += 1
  
  
  if error_counts >= 0 : print(str(error_counts) + ' errors among ' + str(query_counts) + ' query rows, ' + str(value_counts) + ' values.')
  cursor.close()
  conn.close()
  return X.values()

# X = list(get_X_of_the_date(date(2021, 2, 1), device_keys_table[:1], event_keys_table))

#%%
# get Y of the date:
# collect y of the specific [date] and specific device(s) in [device_keys_table] according to specific [time_window]
# y of a device = [whether the device will malfunction in the future between time window]
def get_Y_of_the_date(date: date, device_keys_table, time_window=7, host=mssql_host, user=mssql_user, passward=mssql_password, database=mssql_database):
  conn = pymssql.connect(
    host=host,
    user=user,
    password=passward,
    database=database,
  )
  cursor = conn.cursor(as_dict=True)
  Y = {}
  value_counts = 0
  for device_key in device_keys_table:
    # y = {'device_key':device_key, 'target':0}
    y = {'target':0}
    value_counts += 1
    Y[device_key] = y
    
  key_not_found_counts = 0
  query_counts = 0
  
  cmd_key_constrain = ''
  if len(device_keys_table) <= 15:
    cmd_key_constrain = ' and (concat(svce_loc_id, dev_name) = \'' + device_keys_table[0] + '\''
    for key in device_keys_table[1:]:
      cmd_key_constrain += ' or concat(svce_loc_id, dev_name) = \'' + key + '\''
    cmd_key_constrain += ')'
    
  cmd = ''
  # if len(device_keys_table) == 1:
  #   cmd = """
  #   select concat(svce_loc_id, dev_name) as device_key, count(dev_name) as counts
  #   from ATIM工單_view
  #   where dev_name not like \'%VATIM%\' and 報修日期 between \'{0}\' and \'{1}\' and concat(svce_loc_id, dev_name) = \'{2}\'
  #   group by concat(svce_loc_id, dev_name)
  #   """.format(str(date + timedelta(days=1)), str(date + timedelta(days=time_window + 1)), device_keys_table[0])
  # else:
  cmd = """
  select concat(svce_loc_id, dev_name) as device_key, count(dev_name) as counts
  from ATIM工單_view
  where dev_name not like \'%VATIM%\' and 報修日期 between \'{0}\' and \'{1}\'{2}
  group by concat(svce_loc_id, dev_name)
  """.format(str(date + timedelta(days=1)), str(date + timedelta(days=time_window + 1)), cmd_key_constrain)
  cursor.execute(cmd)  
  for row in cursor:
    # query_counts += 1
    if row['device_key'] in Y:
      Y[row['device_key']]['target'] = int(row['counts'])
    # else: key_not_found_counts += 1

  # if key_not_found_counts > 0 : print(str(key_not_found_counts) + ' key not founds among ' + str(query_counts) + ' query rows, ' + str(value_counts) + ' values.')
  cursor.close()
  conn.close()
  return Y.values()

#%%
# local archive reduce data collection time and support variable [from_date], [to_date], [device_keys_table]
def get_XY_between_date(from_date: date, to_date: date, device_keys_table, event_keys_table, week_based=False, time_window_x=7, time_steps=4, time_window_y=7, host=mssql_host, user=mssql_user, passward=mssql_password, database=mssql_database, use_archive=True):
  print('\ncollecting X, Y ...')
  X = []
  Y = []
  save_path = collected_data_folder + 'XY.csv'
  if path.exists(save_path) and use_archive:
    print('found csv archive, collecting from csv ...')
    time_consumn = datetime.datetime.now()
    with open(save_path, newline='') as csvfile:
      rows = csv.reader(csvfile)
      header = next(rows)
      for row in rows:
        x = {}
        if not row[0] in device_keys_table: continue
        if from_date <= datetime.datetime.strptime(row[1], "%Y-%m-%d").date() < to_date:
          if (datetime.datetime.strptime(row[1], "%Y-%m-%d").date() - from_date).days % 7 == 0 or not week_based:
            for i in range(len(header) - 1):
              if i < 2:
                x[header[i]] = row[i]
              else :
                x[header[i]] = int(row[i])
            # x.append(row[0])
            # x.append(row[1])
        else : continue
        X.append(x)
        y = {}
        y[header[-1]] = int(row[-1])
        Y.append(y)
    time_consumn = datetime.datetime.now() - time_consumn
    print('X, Y loaded from csv archive in ' + str(time_consumn.seconds) + ' seconds.')
    return DataFrame(X), DataFrame(Y)
  
  print('csv archive not found, collecting from server ...')
  cursor_date = from_date
  total_dataset_size = (to_date - from_date).days
  collected_dataset_size = 0
  while cursor_date < to_date:
    try:
      X_new = get_X_of_the_date(cursor_date, 
                                device_keys_table, 
                                event_keys_table, 
                                time_window=time_window_x, 
                                time_steps=time_steps,
                                host=host, 
                                user=user, 
                                passward=passward,
                                database=database)
      Y_new = get_Y_of_the_date(cursor_date, 
                                device_keys_table, 
                                time_window=time_window_y,
                                host=host, 
                                user=user, 
                                passward=passward,
                                database=database)
    except:
      print('\nDB-Lib error message 20047, DBPROCESS is dead or not enabled, resending query ...\n')
      continue
    X.extend(X_new)
    Y.extend(Y_new)
    collected_dataset_size += 1
    print('[' + str(datetime.datetime.now()) + '] ' + str(collected_dataset_size) + ' blocks collected among ' + str(total_dataset_size))
    if week_based:
      cursor_date = cursor_date + datetime.timedelta(days=7)
    else:
      cursor_date = cursor_date + datetime.timedelta(days=1)
  print('X, Y loaded from server.')
  
  print('saving X, Y archive ...')
  with open(save_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    x = list(X[0].keys())
    y = list(Y[0].keys())
    row = []
    row.extend(x)
    row.extend(y)
    writer.writerow(row)
    # writer.writerow(list(X[0].keys()).extend(list(Y[0].keys())))
    for i in range(len(X)):
      # row = list(X[i].values()).extend(list(Y[i].values()))
      # print(row)
      x = list(X[i].values())
      y = list(Y[i].values())
      row = []
      row.extend(x)
      row.extend(y)
      writer.writerow(row)
  print('X, Y archived.')
  return DataFrame(X), DataFrame(Y)

# %%
import numpy as np
def data_argument(X:DataFrame, Y:DataFrame, multiply_y=1):
  if multiply_y > 1:
    print('argumenting records that y > 0 ...')
    # counts_y1 = 0
    
    argumented_XY = DataFrame.copy(X)
    argumented_XY['target'] = Y['target']
    
    argumented_XY = argumented_XY[argumented_XY['target'] > 0]
    
    
    argumented_Y = DataFrame(argumented_XY.pop('target'))
    argumented_X = argumented_XY
    for i in range(multiply_y - 1):
      X = X.append(argumented_X, ignore_index=True)
      Y = Y.append(argumented_Y, ignore_index=True)
    print('data argumented.')#, ' + str(counts_y1 * multiply_y) + ' records that y > 0 .')
    print('Argumented X_train_df.shape = ' + str(X.shape) + '\tArgumented Y_train_df.shape = ' + str(Y.shape))
    return X, Y

# %%
def convert_to_model_input(X:DataFrame, Y:DataFrame, muliply_input_by=1, clamp_y=True):
  X_converted = X.copy()
  Y_converted = Y.copy()
  
  X_converted.pop('device_key')
  X_converted.pop('date')
  X_converted = np.asarray(X_converted).astype('float32')

  Y_converted = np.asarray(Y_converted).astype('float32')
  if clamp_y:
    Y_converted = np.where(Y_converted > 0, 1, 0)
    
  if muliply_input_by == 1:
    return X_converted, Y_converted
  else :
    X_converted_list = []
    Y_converted_list = []
    for i in range(muliply_input_by):
      X_converted_list.append(X)
      Y_converted_list.append(Y)
    return X_converted_list, Y_converted_list
  
def convert_to_model_input_X(X:DataFrame, muliply_input_by=1):
  X.pop('device_key')
  X.pop('date')
  X_np = np.asarray(X).astype('float32')
    
  if muliply_input_by == 1:
    return X_np
  else :
    X_list = []
    for i in range(muliply_input_by):
      X_list.append(X_np)
    return X_list

#%%
# rough safe range of predict_date : 3/10 - 4/19
def get_practical_XY_train_test_of_date(predict_date:date, device_keys_table, event_keys_table, trace_back_to_week=4, time_window_x=2, time_steps=7, time_window_y=7):
  train_from_date = predict_date - timedelta(weeks=trace_back_to_week)
  test_to_date = predict_date + timedelta(days=1)
  X_train_df, Y_train_df = get_XY_between_date(train_from_date, 
                                        predict_date, 
                                        device_keys_table, 
                                        event_keys_table,
                                        time_window_x=time_window_x, 
                                        time_steps=time_steps, 
                                        time_window_y=time_window_y,
                                        use_archive=True)
  X_test_df, Y_test_df = get_XY_between_date(predict_date, 
                                        test_to_date, 
                                        device_keys_table, 
                                        event_keys_table,
                                        time_window_x=time_window_x, 
                                        time_steps=time_steps, 
                                        time_window_y=time_window_y,
                                        use_archive=True)
  print('X_train_df.shape = ' + str(X_train_df.shape) + '\tY_train_df.shape = ' + str(Y_train_df.shape))
  print('X_test_df.shape = ' + str(X_test_df.shape) + '\tY_test_df.shape = ' + str(Y_test_df.shape))
  return X_train_df, Y_train_df, X_test_df, Y_test_df

#%%
# rough safe range of predict_date : 3/10 - 4/19
def get_practical_XY_train_of_date(predict_date:date, device_keys_table, event_keys_table, trace_back_to_week=4, time_window_x=2, time_steps=7, time_window_y=7):
  train_from_date = predict_date - timedelta(weeks=trace_back_to_week)
  train_to_date = predict_date - timedelta(days=time_window_y)
  X_train_df, Y_train_df = get_XY_between_date(train_from_date, 
                                        train_to_date, 
                                        device_keys_table, 
                                        event_keys_table,
                                        time_window_x=time_window_x, 
                                        time_steps=time_steps, 
                                        time_window_y=time_window_y,
                                        use_archive=True)
  print('X_train_df.shape = ' + str(X_train_df.shape) + '\tY_train_df.shape = ' + str(Y_train_df.shape))
  return X_train_df, Y_train_df

#%%