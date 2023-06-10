import pandas as pd
import tensorflow as tf
import datetime
import numpy as np

INTRO_SCREEN_TEXT = """
     _______..______   ___   ___ .___________..______          ___       _______   _______ .______      
    /       ||   _  \  \  \ /  / |           ||   _  \        /   \     |       \ |   ____||   _  \     
   |   (----`|  |_)  |  \  V  /  `---|  |----`|  |_)  |      /  ^  \    |  .--.  ||  |__   |  |_)  |    
    \   \    |   ___/    >   <       |  |     |      /      /  /_\  \   |  |  |  ||   __|  |      /     
.----)   |   |  |       /  .  \      |  |     |  |\  \----./  _____  \  |  '--'  ||  |____ |  |\  \----.
|_______/    | _|      /__/ \__\     |__|     | _| `._____/__/     \__\ |_______/ |_______|| _| `._____|
"""

class Stock:
    def __init__(self, stock_id):
        """
        Initialize a Stock instance with a given stock_id.
        Fetches stock data for the instance.
        """
        self.id = stock_id
        self.get_stock_data()

    def get_stock_data(self):
        """
        Fetches stock data for the stock instance using Yahoo Finance API.
        """
        start_period, end_period = get_epoch_time()
        yahoo_api_url = f"https://query1.finance.yahoo.com/v7/finance/download/{self.id}?period1={start_period}&period2={end_period}&interval=1d&events=history"
        try:
            self.data = pd.read_csv(yahoo_api_url)
        except Exception:
            print("Invalid stock id")
            exit()

def get_epoch_time():
    """
    Calculates the start and end periods for fetching stock data from Yahoo Finance API.
    Returns the start and end periods as epoch timestamps.
    """
    start_period = int((datetime.datetime.now() - datetime.timedelta(days=8)).timestamp())
    end_period = int((datetime.datetime.now() - datetime.timedelta(days=1)).timestamp())
    return start_period, end_period

def predict_stocks(stocks_map, prediction_model):
    """
    Predicts the best stock to buy from a list of Stock instances using a prediction model.
    Returns the best stock based on the prediction model's output.
    """
    best_stock = stocks_map[0]  # Initialize the best_stock variable with the first stock in the stocks_map list

    for stock in stocks_map:  # Iterate over each stock in the stocks_map list
        # Calculate the price change ratio for each stock
        stock.data['Ratio'] = (stock.data['Close'] / stock.data['Open']) - 1
        stock.price_changes = np.reshape(stock.data['Ratio'].values, (-1, 1))
        stock.prediction = prediction_model.predict(stock.price_changes)  # Predict the stock's price changes using the prediction model

        # Get the last prediction value
        last_prediction = stock.prediction[-1][0]  # Retrieve the last predicted value from the stock's prediction
        if last_prediction > best_stock.prediction[-1][0]:  # Check if the last prediction is greater than the best_stock's prediction
            best_stock = stock  # Update the best_stock variable if the current stock has a higher prediction

    return best_stock  # Return the stock with the highest prediction value as the best stock

def print_predictions(stocks_map):
    """
    Prints the predictions for each stock in the stocks_map.
    """
    for stock in stocks_map:
        print(f"{stock.id}: {stock.prediction}%")

def get_user_stocks():
    """
    Prompts the user to input stock ids and returns a list of stock ids.
    """
    stocks_list = []

    while True:
        stocks_list.append(input("Insert stock id: "))
        if 'y' != input("would you like to predict another stock? (y/n): "):
            break

    return stocks_list

def main():
    """
    Main function for running the stock prediction program.
    """
    model_location = "/home/ubuntu/Documents/projects/spxrnnv2.h5"
    prediction_model = tf.keras.models.load_model(model_location)
    stocks_map = []  # Create an empty list to store stock instances

    print(INTRO_SCREEN_TEXT)  # Print the intro screen text

    # Ask user for input
    stocks_list = get_user_stocks()  # Call a function to get user input for the list of stocks

    # Create stock instances
    for stock_id in stocks_list:  # Iterate over each stock ID in the stocks_list
        stocks_map.append(Stock(stock_id))  # Create a new Stock instance with the current stock ID and add it to the stocks_map list

    # Analyze stocks
    best_stock = predict_stocks(stocks_map, prediction_model)  # Call the predict_stocks function to analyze the stocks and get the best stock

    # Print predictions
    print(f"Best stock to buy: {best_stock.id}")  # Print the ID of the best stock to buy

if __name__ == "__main__":
    main()
