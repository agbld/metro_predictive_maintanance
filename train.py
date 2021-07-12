#%%
from matplotlib import use
import get_dataset
from datetime import date
import numpy as np
from pandas import DataFrame

device_keys_table = get_dataset.get_device_keys_table(use_archive=True)
event_keys_table = get_dataset.get_event_keys_table(use_archive=True)

time_steps = 4
time_window_x = 7
time_window_y = 7
X, Y = get_dataset.get_XY_between_date(date(2021, 1, 29), 
                                       date(2021, 4, 19), 
                                       device_keys_table, 
                                       event_keys_table,
                                       time_window_x=time_window_x, 
                                       time_steps=time_steps, 
                                       time_window_y=time_window_y,
                                       use_archive=True)

# X = np.asarray(X).astype('float32')
# X = np.array(X.reshape((len(X), time_steps, 352, 1)))

# Y = np.asarray(Y).astype('float32')
# Y = np.where(Y > 0, 1, 0)

# print(X.shape)
# print(Y.shape)


# %%
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from model_bench import bench_model, test_model, plot_result

num_of_features = len(X[0])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

num_of_fit = 1
num_of_epochs = 100
nodes_of_each_layer = [
      4 * num_of_features, 
      4 * num_of_features,
]
model_name = 'model_Conv2D_x4_1'

# model = Sequential(name=model_name)
# model.add(Dense(num_of_features * 4, activation='relu', input_shape=(num_of_features,)))
# model.add(Dense(num_of_features * 4, activation='relu', input_shape=(num_of_features,)))
# model.add(Dense(1, activation='sigmoid'))
# opt = Adam(lr=1e-4, decay=1e-4)
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# # mean_absolute_error
# # binary_crossentropy
# print(model.summary())

model = Sequential(name=model_name)
model.add(Conv2D(filters=5, kernel_size=(time_steps, 1), activation='relu', input_shape=(time_steps, 352, 1), padding='valid'))
# model.add(Conv2D(filters=256, kernel_size=(1, 352), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
opt = Adam(lr=1e-4/2, decay=1e-4/2)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# mean_absolute_error
# binary_crossentropy

print(model.summary())

bench_result = bench_model(model, num_of_epochs, X_train, Y_train, X_test, Y_test, model_name)

plot_result(bench_result, model_name)

#%%
test_model('bench_model_backup/' + model_name, X_test, Y_test)

# %%

#%%
X, Y = get_dataset.get_XY_between_date(date(2021, 1, 29), date(2021, 4, 19), device_keys_table, event_keys_table, time_window_x=7, time_steps=time_steps, time_window_y=7)

X, Y = DataFrame(X), DataFrame(Y)

X = np.asarray(X).astype('float32')
X = np.array(X.reshape((len(X), time_steps, 352, 1)))

Y = np.asarray(Y).astype('float32')
Y = np.where(Y > 0, 1, 0)

print(X.shape)
print(Y.shape)