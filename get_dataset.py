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
  cmd_from_view = "SELECT svce_loc_id, dev_name FROM ATIM工單_view where dev_name not like '%VATIM%' group by svce_loc_id, dev_name"
  cursor.execute(cmd_from_view)
  for row in cursor:
    svce_loc_id = str(row['svce_loc_id']).zfill(3)
    if len(svce_loc_id_list) == 0:
      key_table.append(svce_loc_id + row['dev_name'])
    elif svce_loc_id in svce_loc_id_list:
      key_table.append(svce_loc_id + row['dev_name'])
  
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
def get_XY_between_date(from_date: date, to_date: date, device_keys_table, event_keys_table, time_window_x=7, time_steps=4, time_window_y=7, host=mssql_host, user=mssql_user, passward=mssql_password, database=mssql_database, use_archive=True):
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
        y[header[-1]] = row[-1]
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