import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# General class that can handle different stocks, rolling windows and z-scores
# to calculate, retrieve and display anomaly trading days.

class StockAnomaly:
    def __init__(self,
                 stocks:list[str] = None,
                 history:str = None,
                 start:datetime|str = None,
                 end:datetime|str = None):
        
        # Set stocks used in our research project if none are provided
        self.stocks = ["AMD", "ASML", "GOOG", "META", "NVDA"] if not stocks else stocks
        self.history = '5y' if not history else history # 5y default time period to get history of
        self.start = start # Set specific start and end period to get stock information of.
        self.end = end
        self.stocks_data = yf.Tickers(self.stocks) # Initialise the yFinance with stocks provided.

        # Get the history of one of provided timespans.
        if start and end:
            self.stocks_history = self.stocks_data.history(start=self.start, end=self.end, interval="1d")
        else:
            self.stocks_history = self.stocks_data.history(self.history, interval="1d")
        
        self.__stocks_history_df = self.stocks_history.copy() # Create a 'private' dataframe to keep a ground truth of the stock data
        self.did_calculate_anomalies = False # Has the method calculate succesfully been called yet.


    # Calculate anomaly trading days based on window days history, and z-values for price and volume.
    def calculate(self, window = 30, price_z = 2.5, volume_z = 2.5):
        if self.did_calculate_anomalies and window == self.window and price_z == self.price_z and self.volume_z == volume_z:
            # Unusual days are already calculated
            return True
        
        # Hyper parameters we can fine tune
        self.window = window # Rolling window size in days
        
        # Z-scores: 1 = normal, 2 >= unusual, 3 >= outlier (Bollinger bands takes z-score of >= 2)
        self.price_z = price_z # From what z-score is price change an anomaly
        self.volume_z = volume_z # From what z-score is trading volume an anomaly
        stocks_history = self.__stocks_history_df.copy()

        # Calculate anomaly trading days for all provided stocks.
        for internal_stock in self.stocks:
            
            # ================ PRICE ================
            # Calculate day by day closing price change in percentage of stock
            stocks_history['return', internal_stock] = stocks_history['Close', internal_stock].pct_change()
            
            # Calculate rolling mean and standard deviation per stock
            stocks_history['rolling_mean_P', internal_stock] = stocks_history['return', internal_stock].rolling(window).mean()
            stocks_history['rolling_std_P', internal_stock] = stocks_history['return', internal_stock].rolling(window).std()
            
            # Calculate z-score of rolling window
            stocks_history['rolling_z_P', internal_stock] = (
                (stocks_history['return', internal_stock] - stocks_history['rolling_mean_P', internal_stock]) / stocks_history['rolling_std_P', internal_stock]
            )

            # Anomaly trading day if z-score >= prive_z, outlier in price change detected.
            # Based on price alone:
            # stocks_history['unusual', stock] = stocks_history['rolling_z_P', stock] >= price_z

            # ================ VOLUME ================
            stocks_history['rolling_mean_V', internal_stock] = stocks_history['Volume', internal_stock].rolling(window).mean()
            stocks_history['rolling_std_V', internal_stock] = stocks_history['Volume', internal_stock].rolling(window).std()

            stocks_history['rolling_z_V', internal_stock] = (
                (stocks_history['Volume', internal_stock] - stocks_history['rolling_mean_V', internal_stock]) / stocks_history['rolling_std_V', internal_stock]
            )

            # Based on volume alone:
            # stocks_history['unusual', stock] = stocks_history['rolling_z_V', stock] >= volume_z

            # Anomaly trading day if price and volume are outlier based on past {window} days.
            stocks_history['unusual', internal_stock] = (
                (stocks_history['rolling_z_P', internal_stock].abs() >= price_z) 
                &
                (stocks_history['rolling_z_V', internal_stock].abs() >= volume_z)
            )

        self.stocks_history = stocks_history.copy() # Set global stocks_history to df containing all calcualted variables above.
        self.did_calculate_anomalies = True # Method succesfully calcualted anomaly trading days.
        self.__transform_for_output() # Convert the dataframe to right format.
        return True
    
    # Convert multilevel stock history dataframe to a dataframe with anomaly trading days' stock, date and rolling z-score values.
    def __transform_for_output(self):
        stocks_history = self.stocks_history.copy()

        # Convert multilevel df to single level dataframe.
        output = (
            stocks_history
            .stack(level=1, future_stack=True)
        )
        output = output[output['unusual']] # Filter all unusual days
        output = output[['rolling_z_P', 'rolling_z_V']] # Retrieve only rolling values
        
        output = (
            output
            .reset_index()
            .rename(columns={'Ticker': 'stock', 'Date': 'date'}) # Rename columns
            .sort_values(['stock', 'date']) # Sort by stock then date
            .reset_index(drop=True) # reset index
        )
        output.columns.name = "index"

        self.__output_stocks_history = output
        return True
    

    # Retrieve all anomaly trading days in either dataframe or list.
    def get_anomalies(self, stock:str = None, get_dict_dates:bool = False):
        # Check if the anomalies have been calculated yet, else throw exception.
        if not self.did_calculate_anomalies:
            raise Exception("Anomalies of stocks are not calculated. Run `self.calculate()` before running this function.")
        
        output = None

        # If a specific stock was provided, only return anomaly days for that stock, else output data for all calculated stocks.
        if type(stock) == str:
            if stock not in self.stocks:
                raise Exception(f"Stock: {stock} doesn't exist or does not exist in the current calculation.")
            output = self.__output_stocks_history[self.__output_stocks_history["stock"] == stock].reset_index(drop=True)
        else:
            output = self.__output_stocks_history
        

        if get_dict_dates:
            return output.groupby('stock')['date'].agg(list).to_dict() # Provide stock with list of dates instead of all z-score information.
        return output


    # Plot a graph of the closing price, rolling average and draw lines at anomaly trading days of stock(s).
    def display_anomalies(self, stock = None,figsize:tuple = (12,5)):
        # Throw exceptions if the right data wasn't provided or does not yet exist.
        if not self.did_calculate_anomalies:
            raise Exception("Anomalies of stocks are not calculated. Run `self.calculate()` before running this function.")
        
        stocks = self.stocks
        if stock is not None:
            if stock not in self.stocks:
                raise Exception(f"Stock: {stock} doesn't exist or does not exist in the current calculation.")
            stocks = [stock]

        # Plot stock data: price, rolling average and anomalies
        for internal_stock in stocks:
            # Get index of all unusual trading days
            unusual_days = self.stocks_history[self.stocks_history["unusual", internal_stock]].index

            # Plot closing prices
            plt.figure(figsize=figsize)
            plt.plot(self.stocks_history['Close', internal_stock], 
                    label = "Closing price"
                    )
            
            # Plot rolling average
            plt.plot(
                self.stocks_history['Close', internal_stock].rolling(self.window).mean(),
                label=f"Rolling average price ({self.window} days)"
                )

            # Plot vertical lines at each unusual day
            plt.vlines(
                    unusual_days,
                    ymin=0,
                    ymax=self.stocks_history['Close', internal_stock].max(),
                    color = 'red',
                    linestyle='-', 
                    alpha=0.3, 
                    linewidth=1,
                    label=f'Anomalies (z_P = {self.price_z}, z_V = {self.volume_z})'
                )

            # State dates of anomalies
            for anomaly_day in unusual_days:
                plt.text(anomaly_day, 0, f"{anomaly_day.strftime('%d-%m-%Y')}", rotation = 90)

            plt.legend()
            plt.title(f"{internal_stock}: Unusual Trading Days")
            plt.show()


if '__main__' == __name__:

    print('Running Test')
    
    stock_anomaly = StockAnomaly()

    stock_anomaly.calculate(window=20,price_z=2,volume_z=2)

    print(stock_anomaly.get_anomalies(get_dict_dates=True))

