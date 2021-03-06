#%%
import predictive_maintance
from predictive_maintance import train_model_for_date, make_prediction_of_the_date_with_model, update_archive
from datetime import date, timedelta

# update_archive()

#%%
make_predict_date = date(2021, 4, 1)
# while make_predict_date < date(2021, 4, 2):
model = train_model_for_date(make_predict_date, 'CU307', epochs=200, use_archive=False)

#%%
result_df = make_prediction_of_the_date_with_model(make_predict_date, 'trained_model/model_' + str(make_predict_date), device_type='CU307', use_archive=False)

# %%