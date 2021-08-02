#%%
import predictive_maintance
from predictive_maintance import train_model_for_date, make_prediction_of_the_date_with_model, update_archive
from datetime import date, timedelta


#%%
make_predict_date = date(2021, 3, 1)
while make_predict_date < date(2021, 3, 5):
  model = train_model_for_date(make_predict_date, 'CU307', epochs=200)
  result_df = make_prediction_of_the_date_with_model(make_predict_date, 'trained_model/' + 'model_2021-03-01', device_type='CU307')
  make_predict_date = make_predict_date + timedelta(days=1)
# %%
