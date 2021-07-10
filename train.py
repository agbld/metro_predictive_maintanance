#%%
import get_dataset
from datetime import date
import numpy as np
from pandas import DataFrame

device_keys_table = get_dataset.get_device_keys_table()
event_keys_table = get_dataset.get_event_keys_table()
X, Y = get_dataset.get_XY_between_date(date(2021, 1, 29), date(2021, 2, 23), device_keys_table, event_keys_table)

X, Y = DataFrame(X), DataFrame(Y)

X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('float32')
Y = np.where(Y > 0, 1, 0)

print(X.shape)
print(Y.shape)

# %%
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

num_of_features = len(X[0])

def generate_model():
      model = Sequential()
      model.add(Dense(num_of_features, activation='relu', input_shape=(num_of_features,)))
      model.add(Dense(num_of_features * 2, activation='relu'))
      model.add(Dense(num_of_features, activation='relu'))
      model.add(Dense(1, activation='sigmoid'))

      opt = Adam(lr=1e-4, decay=1e-6 / 200)
      model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
      # mean_absolute_error
      # binary_crossentropy

      print(model.summary())
      
      return model


#%%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
def bench_model(num_of_fit:int, epochs:int, X_train=X_train, Y_train=Y_train):
      loss_list = []
      val_loss_list = []
      accuracy_list = []
      val_accuracy_list = []
      for i in range(epochs):
            loss_list.append(0)
            val_loss_list.append(0)
            accuracy_list.append(0)
            val_accuracy_list.append(0)
      
      for i in range(num_of_fit):
            tmp_model = generate_model()
            history = tmp_model.fit(X_train, Y_train, epochs=epochs, validation_split=0.2, shuffle=True)
            tmp_model.save('trained_model')
            for i in range(epochs):
                  loss_list[i] += history.history['loss'][i]
                  val_loss_list[i] += history.history['val_loss'][i]
                  accuracy_list[i] += history.history['accuracy'][i]
                  val_accuracy_list[i] += history.history['val_accuracy'][i]
      
      for i in range(epochs):
            loss_list[i] /= num_of_fit
            val_loss_list[i] /= num_of_fit
            accuracy_list[i] /= num_of_fit
            val_accuracy_list[i] /= num_of_fit
      return loss_list, val_loss_list, accuracy_list, val_accuracy_list

num_of_fit = 1
num_of_epochs = 15
loss_list, val_loss_list, accuracy_list, val_accuracy_list = bench_model(num_of_fit, num_of_epochs)

from matplotlib import pyplot as plt

plt.plot(loss_list)
plt.plot(val_loss_list)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(accuracy_list)
plt.plot(val_accuracy_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#%%
def test_model(X = X_test[:50]):
      model = load_model('trained_model')
      sample = X
      predictions = model.predict(sample)
      avg_diff = 0
      for i in range(0, len(sample)):
            pred = round(predictions[i][0], 2)
            y = Y_test[i][0]
            diff = abs(y - pred)
            avg_diff += diff
            print(str(i) + 'th' + '\tpred:' + str(pred) + '\tY:' + str(y) + '\tdiff:' + str(diff))
      print(round(avg_diff / len(sample), 2))
# %%