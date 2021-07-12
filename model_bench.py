from tensorflow.keras.models import clone_model

def bench_model(model, epochs:int, X_train, Y_train, X_test, Y_test, model_name='trained_model'):
  history = model.fit(X_train, Y_train, epochs=epochs, validation_split=0.2, shuffle=True)
  model.evaluate(X_test, Y_test)
  model.save('bench_model_backup/' + model_name)
  
  loss_list = history.history['loss']
  val_loss_list = history.history['val_loss']
  accuracy_list = history.history['accuracy']
  val_accuracy_list = history.history['val_accuracy']
  
  return [loss_list, val_loss_list, accuracy_list, val_accuracy_list]
  
def batch_bench_models(model_list, model_names, epochs:int, X_train, Y_train, X_test, Y_test):
  if len(model_list) != len(model_names): 
    print('Ilegal input !')
    return []
  result = []
  for i in range(len(model_list)):
    result.append(bench_model(model_list[i],
                              epochs, 
                              X_train,
                              Y_train,
                              X_test, 
                              Y_test,
                              'batch_bench_model_backup/' + model_names[i]))
  return result

from matplotlib import pyplot as plt
def plot_result(bench_result, model_name):
  plt.plot(bench_result[0])
  plt.plot(bench_result[1])
  plt.title(model_name + ' loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()

  plt.plot(bench_result[2])
  plt.plot(bench_result[3])
  plt.title(model_name + ' accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()

from tensorflow.keras.models import load_model
import csv
def test_model(model_name, X_test, Y_test):
  model = load_model(model_name)
  sample = X_test
  predictions = model.predict(sample)
  correctness = 0
  pred_0 = 0
  pred_1 = 0
  correctness_pred_0 = 0
  correctness_pred_1 = 0
  
  with open('collected_data/test_result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['pred_raw', 'pred', 'y', 'diff'])
    for i in range(0, len(sample)):
      pred_raw = round(predictions[i][0], 2)
      pred = 0
      if pred_raw > 0.5 : pred = 1
      y = Y_test[i][0]
      diff = abs(y - pred)
      correctness += 1 - diff
      pred_0 += 1 - pred
      pred_1 += pred
      if pred == 0 and pred == y: correctness_pred_0 += 1
      if pred == 1 and pred == y: correctness_pred_1 += 1
      row = [pred_raw, pred, y, diff]
      writer.writerow(row)
      print(str(i) + 'th' + '\tpred_raw:' + str(pred_raw) + '\tpred:' + str(pred) + '\tY:' + str(y) + '\tdiff:' + str(diff))
  correctness = round(correctness / (pred_0 + pred_1), 4) * 100
  correctness_pred_0 = round(correctness_pred_0 / pred_0, 4) * 100
  correctness_pred_1 = round(correctness_pred_1 / pred_1, 4) * 100
  print('correctness: ' + str(correctness) + '%\tcorrectness(pred:0): ' + str(correctness_pred_0) + '%\tcorrectness(pred:1): ' + str(correctness_pred_1) + '%')
      