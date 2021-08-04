#%%
import os
import pandas as pd
from pandas import DataFrame
import csv
from tqdm import tqdm
import sqlalchemy
from sqlalchemy import event
import pyodbc
from urllib.parse import quote_plus

from data_transform import chunker
import data_transform

driver_name = '{ODBC Driver 17 for SQL Server}'

def upload_csv_to_mssql(conn, from_folder: str, to_table: str, export_csv = False):
  from_folder = from_folder + '/'
  result_path = from_folder + '/result.csv'
  onlyfiles = []
  for f in os.listdir(from_folder):
      if os.path.isfile(os.path.join(from_folder, f)): onlyfiles.append(f)
  num_of_files = len(onlyfiles)

  result_list = []
  for file in onlyfiles:
    print("uploading {} to sql server...".format(str(file)))
    df = DataFrame(pd.read_csv(from_folder + file, delimiter=';', encoding='utf8'))
    df.pop('station_name')
    df.pop('logical_id')
    df.pop('device_id')
    df.pop('msg_level')
    df.pop('desc_english')
    df.pop('desc_chinese')
    result_list.append(df)
    chunk_size = 10000
    chunked_df = chunker(df, chunk_size)
    chunked_df_list = []
    with tqdm(total=int(df.shape[0] / chunk_size + 0.9)) as pbar:
      for chunk in chunked_df:
        chunk.to_sql(name=to_table, if_exists="append", con=conn, index=False)
        pbar.update(1)
    os.rename(from_folder + file, from_folder + 'uploaded/' + file)
  if export_csv:
    result = pd.concat(result_list)
    result.to_csv(result_path)

if __name__ == '__main__':
  mssql_host, mssql_user, mssql_password, mssql_database = '', '', '', ''
  with open('refer/mssql_info.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    mssql_host=next(rows)[0]
    mssql_user=next(rows)[0]
    mssql_password=next(rows)[0]
    mssql_database=next(rows)[0]
  
  conn =  "DRIVER={0};".format(driver_name) + "SERVER={0};DATABASE={1};UID={2};PWD={3}".format(mssql_host, mssql_database, mssql_user, mssql_password)
  quoted = quote_plus(conn)
  new_con = 'mssql+pyodbc:///?odbc_connect={}'.format(quoted)
  conn = sqlalchemy.create_engine(new_con)
  
  @event.listens_for(conn, 'before_cursor_execute')
  def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
      # print("FUNC call")
      if executemany:
          cursor.fast_executemany = True
  
  upload_csv_to_mssql(conn, 'ATIM_event_alarm', data_transform.atim_event_alarm_table)
# %%
