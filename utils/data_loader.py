import pandas as pd

def load_data(ticker):
    file_path = f'data/{ticker}_processed_data.csv'
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return data