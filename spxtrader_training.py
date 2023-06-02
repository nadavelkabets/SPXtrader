import pandas as pd
import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math

HISTORY_DAYS = 1000
TIME_STEPS = 5

class Stock:
  def __init__(self, stock_id):
    self.id = stock_id
    self.get_stock_data()

  def get_stock_data(self):
      start_period, end_period = get_epoch_time()
      yahoo_api_url = f"https://query1.finance.yahoo.com/v7/finance/download/{self.id}?period1={start_period}&period2={end_period}&interval=1d&events=history"
      self.data = pd.read_csv(yahoo_api_url)

def get_epoch_time():
  #1000 days of data
  start_period = int((datetime.datetime.now() - datetime.timedelta(days=HISTORY_DAYS)).timestamp())
  end_period = int((datetime.datetime.now() - datetime.timedelta(days=1)).timestamp())

  return start_period, end_period

def get_stocks(spx_stocks):
  stocks_map = []

  #create stock instance
  #try except will skip unavilable data
  for stock_id in spx_stocks:
    try:
      stocks_map.append(Stock(stock_id))
    except Exception:
      print(f"{stock_id} not found")

  return stocks_map

def process_stocks_data(stocks_map):
  #returns a list of (lists of historic stock prices)
  stock_prices_by_stock = []

  for stock in stocks_map:
    stock.data['Ratio'] = (stock.data['Close'] / stock.data['Open'] - 1) * 10
    stock_prices_by_stock.append(stock.data['Ratio'].tolist())
  
  return stock_prices_by_stock

def create_training_data(stock_prices_by_stock):
  #splits the list of historic stock prices to train batches of 5 days
  five_day_train_data = []
  day_six_output_data = []
  # running on the list of stocks
  for stock_data in stock_prices_by_stock:
    for batch_index in range(len(stock_data) - TIME_STEPS):
      five_day_train_data.append(stock_data[batch_index : batch_index + TIME_STEPS])
      day_six_output_data.append(stock_data[batch_index + TIME_STEPS])

  return five_day_train_data, day_six_output_data

def create_training_dataset(five_day_train_data, day_six_output_data, batch_size):
  #combine inputs and outputs into a single dataset
  dataset = list(zip(five_day_train_data, day_six_output_data))
  #shuffle the dataset
  np.random.shuffle(dataset)
  #separate inputs and outputs
  shuffled_five_day_train_data, shuffled_day_six_output_data = zip(*dataset)
  #convert to numpy arrays
  shuffled_five_day_train_data = np.array(shuffled_five_day_train_data)
  shuffled_day_six_output_data = np.array(shuffled_day_six_output_data)
  #reshape the inputs
  shuffled_five_day_train_data = np.reshape(shuffled_five_day_train_data, (shuffled_five_day_train_data.shape[0], TIME_STEPS, 1))
  #return the shuffled and reshaped inputs and outputs
  return shuffled_five_day_train_data, shuffled_day_six_output_data


def filter_data(five_day_train_data, day_six_output_data):
  filtered_five_day_train_data = []
  filtered_day_six_output_data = []
    
  for batch_index in range(len(five_day_train_data)):
    if 0 in five_day_train_data[batch_index] or any(math.isnan(x) for x in five_day_train_data[batch_index])\
      or 0 == day_six_output_data[batch_index] or math.isnan(day_six_output_data[batch_index]):
      continue
      
    filtered_five_day_train_data.append(five_day_train_data[batch_index])
    filtered_day_six_output_data.append(day_six_output_data[batch_index])
    
  return filtered_five_day_train_data, filtered_day_six_output_data

def main():
    spx_components_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    spx_components_table = pd.read_csv(spx_components_url)
    spx_stock_names = spx_components_table["Symbol"].tolist()
    
    #get stock data
    stocks_map = get_stocks(spx_stock_names)
    #process data
    stock_prices = process_stocks_data(stocks_map)
    #split data to batches
    five_day_train_data, day_six_output_data = create_training_data(stock_prices)
    #filter NaN and 0s in the dataset
    five_day_train_data, day_six_output_data = filter_data(five_day_train_data, day_six_output_data)
    #create training dataset
    batch_size = 5
    shuffled_five_day_train_data, shuffled_day_six_output_data = create_training_dataset(five_day_train_data, day_six_output_data, batch_size)
   
    #RNN model
    stock_prediction_model = tf.keras.Sequential([
      tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(TIME_STEPS, 1)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.LSTM(128, return_sequences=True),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.LSTM(64),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(1)
    ])
    #compile the model
    stock_prediction_model.compile(optimizer='adam', loss='mean_absolute_error')
    #train with GPU
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
      raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
    #train model
    epoch_count = 10
    with tf.device('/GPU:0'):
      stock_prediction_model.fit(shuffled_five_day_train_data, shuffled_day_six_output_data, epochs=epoch_count)
    #save model
    stock_prediction_model.save("stock_prediction_model")
   
if __name__ == "__main__":
    main()
