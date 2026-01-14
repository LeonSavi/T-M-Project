import yfinance as yf
import matplotlib.pyplot as plt

class StockAnomaly:
    def __init__(self, stocks, history:str = "5y", start = None, end = None):
        self.stocks = stocks
        self.history = history
        self.start = start
        self.end = end
        self.stocks_data = yf.Tickers(self.stocks)

        if start and end:
            self.stocks_history = self.stocks_data.history(start=start, end=end, interval="1d")
        else:
            self.stocks_history = self.stocks_data.history(history, interval="1d")
        
        self.__stocks_history_df = self.stocks_history.copy()
        self.did_calculate_anomalies = False



    def calculate(self, window = 30, price_z = 2.5, volume_z = 2.5):
        if self.did_calculate_anomalies and window == self.window and price_z == self.price_z and self.volume_z == volume_z:
            # Unusual days are already calculated
            return True
        
        # Hyper parameters we can fine tune
        self.window = window # Rolling window size in days
        # Z scores: 1 = normal, 2 >= unusual, 3 >= outlier
        self.price_z = price_z # From what z-score is price change an anomaly
        self.volume_z = volume_z # From what z-score is trading volume an anomaly
        stocks_history = self.__stocks_history_df.copy()


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

            # Anomaly trading day if z-score > 3, outlier in price change detected.
            # Based on price alone:
            # stocks_history['unusual', stock] = stocks_history['rolling_z_P', stock] > 3

            # ================ VOLUME ================
            stocks_history['rolling_mean_V', internal_stock] = stocks_history['Volume', internal_stock].rolling(window).mean()
            stocks_history['rolling_std_V', internal_stock] = stocks_history['Volume', internal_stock].rolling(window).std()

            stocks_history['rolling_z_V', internal_stock] = (
                (stocks_history['Volume', internal_stock] - stocks_history['rolling_mean_V', internal_stock]) / stocks_history['rolling_std_V', internal_stock]
            )

            # Based on volume alone
            # stocks_history['unusual', stock] = stocks_history['rolling_z_V', stock] > 3

            # Anomaly trading day if price and volume are outlier based on past 30 days.
            stocks_history['unusual', internal_stock] = (
                (stocks_history['rolling_z_P', internal_stock].abs() >= price_z) 
                &
                (stocks_history['rolling_z_V', internal_stock].abs() >= volume_z)
            )

        self.stocks_history = stocks_history.copy()
        self.did_calculate_anomalies = True
        self.__transform_for_output()
        return True
    
    def __transform_for_output(self):
        stocks_history = self.stocks_history.copy()

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
    
    def get_anomalies(self, stock = None, get_list = False):
        if not self.did_calculate_anomalies:
            raise Exception("Anomalies of stocks are not calculated. Run calculate() before running this function.")
        
        output = None

        if type(stock) == str:
            if stock not in self.stocks:
                raise Exception(f"Stock: {stock} doesn't exist or does not exist in the current calculation.")
            output = self.__output_stocks_history[self.__output_stocks_history["stock"] == stock].reset_index(drop=True)
        else:
            output = self.__output_stocks_history

        if get_list:
            return output.values.tolist()
        return output


    def display_anomalies(self, stock = None):
        if not self.did_calculate_anomalies:
            raise Exception("Anomalies of stocks are not calculated. Run calculate() before running this function.")
        
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
            plt.figure(figsize=(12,5))
            plt.plot(self.stocks_history['Close', internal_stock], 
                    label = "Closing price"
                    )
            
            # Plot rolling average
            plt.plot(
                self.stocks_history['Close', internal_stock].rolling(self.window).mean(),
                label=f"Rolling average price ({self.window} days)"
                )

            # Plot line at each unusual day
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

    