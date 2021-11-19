import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso

WINDOW_SIZES = [7, 14, 28, 56, 224, 700, 1400]
EWM_COLS = [f'ewm_{ws}' for ws in WINDOW_SIZES]
EWM_DELTA_COLS = [f'ewm_{ws}_delta' for ws in WINDOW_SIZES]
feature_columns = EWM_DELTA_COLS + ['ewm_delta_sum', 'ewm_delta_mean', 'exp_trend_delta']

def get_increase_pct(prices, horizon=30):
    res = np.vstack([prices.shift(-i).values for i in range(horizon)])
    max_value_in_horizon = np.nanmax(res, axis=0)
    return max_value_in_horizon / prices
    

def add_ewm(df, price_col='Close'):
    for ws in WINDOW_SIZES:
        ewm = df[price_col].ewm(span=ws).mean()
        df.loc[:, f'ewm_{ws}'] = ewm


def add_exp_trend(df, price_col='Close'):
    xs = np.array(list(df.index))
    ys = np.log(df[price_col].values)

    model = Lasso()
    model.fit(xs.reshape(-1, 1), ys.reshape(-1, 1))
    preds = model.predict(xs.reshape(-1, 1))

    a = model.coef_[0]
    b = preds[0] - xs[0] * a
    df.loc[:, 'exp_trend'] = np.exp(preds)
    return a, b


def plot(df, cols, dt_col='DT', price_col='Close', buy_column=None, sell_column=None, height=0.0005, name='graph'):
    plt.figure(figsize=(20,10))
    for col in cols:
        plt.plot(df[dt_col], df[col])

    
    if buy_column is not None:
        buy_df = df[df[buy_column].astype(int) > 0]
        plt.vlines(buy_df.DT, buy_df[price_col]*(1-height), buy_df[price_col]*(1+height), colors='green')

    if sell_column is not None:
        sell_df = df[df[sell_column].astype(int) > 0]
        plt.vlines(sell_df.DT, sell_df[price_col]*(1-height), sell_df[price_col]*(1+height), colors='green')

    plt.legend(cols)
    plt.savefig(f'{name}.png', format='png')
    plt.show()


EWM_COLS = [f'ewm_{ws}' for ws in WINDOW_SIZES]
EWM_DELTA_COLS = [f'ewm_{ws}_delta' for ws in WINDOW_SIZES]


class Trader:
    def __init__(self, dt_col='DT', 
                price_col='Close'):

        self.dt_col = dt_col
        self.price_col = price_col
    
    
    def fit(self, price_df):
        target_cols = EWM_COLS + ['exp_trend']
        self.price_df = price_df[[self.dt_col, self.price_col] + target_cols].copy()
        self.price_df.columns = ['DT', 'price'] + target_cols

        self.add_indicators(self.price_df)

    
    def add_indicators(self, df, columns=EWM_COLS+['exp_trend']):
        for col in columns:
            value = df[col]
            delta = value - df.price
            df.loc[:, f'{col}_delta'] = delta / df.price
            
        df.loc[:, 'ewm_delta_sum'] = df[EWM_DELTA_COLS].sum(axis=1)
        df.loc[:, 'ewm_delta_mean'] = df[EWM_DELTA_COLS].mean(axis=1)
        return df
        

    def buy_by_indicator(self, df, column, n_calc=100):

        mean_prices = []
        thresholds = np.linspace(-1, 2, n_calc)
        for threshold in thresholds:
            buy_df = df[df[column] > threshold]
            if buy_df.shape[0] > 0:
                mean_prices.append(buy_df.price.mean())
            else:
                mean_prices.append(np.inf)
        best_price_ind = np.argmin(mean_prices)
        return mean_prices[best_price_ind], thresholds[best_price_ind]
