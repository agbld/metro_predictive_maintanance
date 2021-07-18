#%%
import predictive_maintance
from predictive_maintance import train_model_for_date, make_prediction_of_the_date_with_model
from datetime import date, timedelta

#%%
make_predict_date = date(2021, 3, 1)
while make_predict_date < date(2021, 3, 2):
  model = train_model_for_date(make_predict_date, 'CU307', epochs=5)
  result_df = make_prediction_of_the_date_with_model(make_predict_date, 'trained_model/' + model.name, device_type='CU307')
  make_predict_date = make_predict_date + timedelta(days=1)
# %%
