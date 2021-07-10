#%%
from os import path
import datetime
from datetime import date, timedelta
import csv
import pymssql
from scipy.sparse import data

event_key_table_path = 'refer/事件警報對照表_toAFC.csv'
collected_data_folder = 'collected_data/'
mssql_host='wfx1.duckdns.org'
mssql_user='agiblida'
mssql_password='lc@32117'
mssql_database='metro_maintenance'

#%%
def get_device_keys_table(host=mssql_host, user=mssql_user, passward=mssql_password, database=mssql_database, svce_loc_id_list=[]):
  print('\ncollecting device key table ...')
  key_table = [] # the device key table
  save_path = collected_data_folder + 'device_keys_table.csv'
  if path.exists(save_path):
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
def get_event_keys_table(csv_path=event_key_table_path, host=mssql_host, user=mssql_user, passward=mssql_password, database=mssql_database):
  print('\ncollecting event key table ...')
  key_table = [] # the event key table
  save_path = collected_data_folder + 'event_keys_table.csv'
  if path.exists(save_path):
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
def get_X_of_the_date(date: date, device_keys_table, event_keys_table, host=mssql_host, user=mssql_user, passward=mssql_password, database=mssql_database):
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
    # x = {'device_key':device_key}
    x = {}
    for event_key in event_keys_table:
      for i in range(1, 5):
        x[event_key + "_" + str(i)] = 0
        value_counts += 1
    X[device_key] = x
    
  error_counts = 0
  query_counts = 0
  
  cmd = ""
  for i in range(1, 5):
    cmd += """
    select svce_loc_id, dev_name, concat(tag_name, param_value) as event_key, \'{0}\' as week, count(concat(tag_name, param_value)) as counts
    from [ATIM_event_alarm]
    where create_datetime between \'{1}\' and \'{2}\' and dev_name not like \'%VATIM%\'
    group by concat(tag_name, param_value), svce_loc_id, dev_name
    """.format(i, str(date - timedelta(weeks=i)), str(date - timedelta(weeks=i - 1)))
    if i < 4:
      cmd += " union "
  cursor.execute(cmd)
  for row in cursor:
    query_counts += 1
    device_key = str(row['svce_loc_id']).zfill(3) + row['dev_name']
    if device_key in X and (row['event_key'] + "_" + str(i)) in X[device_key]:
      X[device_key][row['event_key'] + "_" + row['week']] = int(row['counts'])
    else: error_counts += 1
      
  if error_counts > 0 : print(str(error_counts) + ' errors among ' + str(query_counts) + ' query rows, ' + str(value_counts) + ' values.')
  cursor.close()
  conn.close()
  return X.values()

#%%
def get_Y_of_the_date(date: date, device_keys_table, host=mssql_host, user=mssql_user, passward=mssql_password, database=mssql_database):
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
  
  cmd = """
  select concat(svce_loc_id, dev_name) as device_key, count(dev_name) as counts
  from ATIM工單_view
  where dev_name not like \'%VATIM%\' and 報修日期 between \'{0}\' and \'{1}\'
  group by concat(svce_loc_id, dev_name)
  """.format(str(date + timedelta(days=1)), str(date + timedelta(days=8)))
  cursor.execute(cmd)  
  for row in cursor:
    query_counts += 1
    if row['device_key'] in Y:
      Y[row['device_key']]['target'] = int(row['counts'])
    else: key_not_found_counts += 1

  if key_not_found_counts > 0 : print(str(key_not_found_counts) + ' key not founds among ' + str(query_counts) + ' query rows, ' + str(value_counts) + ' values.')
  cursor.close()
  conn.close()
  return Y.values()

#%%
def get_Y_class_of_the_date():
  pass

#%%
def get_XY_between_date(from_date: date, to_date: date, device_keys_table, event_keys_table, host=mssql_host, user=mssql_user, passward=mssql_password, database=mssql_database):
  print('\ncollecting X, Y ...')
  X = []
  Y = []
  save_path = collected_data_folder + 'XY.csv'
  if path.exists(save_path):
    print('found csv archive, collecting from csv ...')
    with open(save_path, newline='') as csvfile:
      rows = csv.reader(csvfile)
      for row in rows:
        x = []
        for value in row[:-1]:
          x.append(int(value))
        X.append(x)
        y = [int(row[-1])]
        Y.append(y)
    print('X, Y loaded from csv archive.')
    return X, Y
  
  print('csv archive not found, collecting from server ...')
  cursor_date = from_date
  total_dataset_size = (to_date - from_date).days
  collected_dataset_size = 0
  while cursor_date < to_date:
    X_new = get_X_of_the_date(cursor_date, device_keys_table, event_keys_table)
    Y_new = get_Y_of_the_date(cursor_date, device_keys_table)
    X.extend(X_new)
    Y.extend(Y_new)
    collected_dataset_size += 1
    print('[' + str(datetime.datetime.now()) + '] ' + str(collected_dataset_size) + ' blocks collected among ' + str(total_dataset_size))
    cursor_date = cursor_date + datetime.timedelta(days=1)
  print('X, Y loaded from server.')
  
  with open(save_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
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
  return X, Y

# %%
def get_quick_XY():
  device_keys_table = get_device_keys_table()
  event_keys_table = get_event_keys_table()
  X, Y = get_XY_between_date(date(2021, 2, 1), date(2021, 2, 15), device_keys_table, event_keys_table)
  return X, Y

# %%