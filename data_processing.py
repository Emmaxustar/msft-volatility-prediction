import numpy as np
import pandas as pd
import yfinance as yf


def feature_engineering(data):

    data.columns = [f"{ticker}_{price}" for price, ticker in data.columns]

    data = data.rename(
        columns=lambda x:
            x.replace('^VIX', 'VIX')
             .replace('^VXN', 'VXN')
             .replace('^TNX', 'TNX')
    )

    cols_to_keep = [
        'MSFT_Close',
        'QQQ_Close',
        'SPY_Close',
        'XLK_Close',
        'SOXX_Close',
        'TLT_Close',
        'TNX_Close',
        'VIX_Close',
        'VXN_Close',
        'MSFT_Volume',
        'QQQ_Volume',
        'SPY_Volume'
    ]

    data = data[cols_to_keep]
    df = data.copy()

    # log returns
    df['MSFT_log_return'] = np.log(df['MSFT_Close'] / df['MSFT_Close'].shift(1))
    df['SPY_log_return'] = np.log(df['SPY_Close'] / df['SPY_Close'].shift(1))
    df['QQQ_log_return'] = np.log(df['QQQ_Close'] / df['QQQ_Close'].shift(1))
    df['XLK_log_return'] = np.log(df['XLK_Close'] / df['XLK_Close'].shift(1))
    df['SOXX_log_return'] = np.log(df['SOXX_Close'] / df['SOXX_Close'].shift(1))
    df['TLT_log_return'] = np.log(df['TLT_Close'] / df['TLT_Close'].shift(1))

    # lag returns
    for lag in [1, 3, 5, 10, 15, 21]:
        df[f'MSFT_log_return_lag{lag}'] = df['MSFT_log_return'].shift(lag)

    # rolling volatility
    for window in [5, 21, 63, 126]:
        df[f'MSFT_rolling_vol_{window}'] = df['MSFT_log_return'].rolling(window).std()

    df['SPY_rolling_vol_5'] = df['SPY_log_return'].rolling(5).std()
    df['SPY_rolling_vol_21'] = df['SPY_log_return'].rolling(21).std()
    df['QQQ_rolling_vol_21'] = df['QQQ_log_return'].rolling(21).std()
    df['XLK_rolling_vol_5'] = df['XLK_log_return'].rolling(5).std()
    df['XLK_rolling_vol_21'] = df['XLK_log_return'].rolling(21).std()
    df['SOXX_rolling_vol_21'] = df['SOXX_log_return'].rolling(21).std()
    df['TLT_rolling_vol_21'] = df['TLT_log_return'].rolling(21).std()

    # volume
    df['MSFT_log_volume'] = np.log(df['MSFT_Volume'])
    df['MSFT_volume_ratio'] = df['MSFT_Volume'] / df['MSFT_Volume'].rolling(21).mean()
    df['MSFT_log_volume_ratio'] = np.log(df['MSFT_volume_ratio'])

    # VIX / VXN
    df['VIX_level'] = df['VIX_Close']
    df['VIX_change'] = df['VIX_Close'].pct_change()
    df['VIX_log_change'] = np.log(df['VIX_Close'] / df['VIX_Close'].shift(1))
    df['VIX_lag1'] = df['VIX_Close'].shift(1)
    df['VIX_rolling_mean_21'] = df['VIX_level'].rolling(21).mean()
    df['VIX_rolling_mean_63'] = df['VIX_level'].rolling(63).mean()

    df['VXN_level'] = df['VXN_Close']
    df['VXN_change'] = df['VXN_Close'].pct_change()
    df['VXN_log_change'] = np.log(df['VXN_Close'] / df['VXN_Close'].shift(1))
    df['VXN_lag1'] = df['VXN_Close'].shift(1)
    df['VXN_rolling_mean_21'] = df['VXN_level'].rolling(21).mean()

    # TNX
    df['TNX_level'] = df['TNX_Close']
    df['TNX_change'] = df['TNX_Close'].pct_change()
    df['TNX_lag1'] = df['TNX_Close'].shift(1)

    # relative performance
    df['MSFT_minus_XLK_return'] = df['MSFT_log_return'] - df['XLK_log_return']
    df['MSFT_minus_SPY_return'] = df['MSFT_log_return'] - df['SPY_log_return']
    df['QQQ_minus_SPY_return'] = df['QQQ_log_return'] - df['SPY_log_return']

    # lagged volatility
    df['MSFT_vol_5_lag1'] = df['MSFT_rolling_vol_5'].shift(1)
    df['MSFT_vol_21_lag1'] = df['MSFT_rolling_vol_21'].shift(1)
    df['MSFT_vol_63_lag1'] = df['MSFT_rolling_vol_63'].shift(1)

    # volatility regime
    df['vol_ratio_5_21'] = df['MSFT_rolling_vol_5'] / df['MSFT_rolling_vol_21']
    df['vol_ratio_21_63'] = df['MSFT_rolling_vol_21'] / df['MSFT_rolling_vol_63']
    df['VIX_relative'] = df['VIX_level'] / df['VIX_rolling_mean_21']

    # target variables
    df['target_volatility_1d'] = df['MSFT_log_return'].shift(-1).abs()
    df['target_volatility_5d'] = df['MSFT_log_return'].shift(-1).rolling(5).std().shift(-4)
    df['target_volatility_21d'] = df['MSFT_log_return'].shift(-1).rolling(21).std().shift(-20)

    # earnings dates
    msft = yf.Ticker("MSFT")
    earnings_dates = msft.earnings_dates.index.tz_localize(None)

    df["earnings_window"] = 0

    for d in earnings_dates:
        mask = (
            (df.index >= d - pd.Timedelta(days=2)) &
            (df.index <= d + pd.Timedelta(days=2))
        )
        df.loc[mask, "earnings_window"] = 1

    core_feature_cols = [
        'MSFT_log_return',
        'MSFT_log_return_lag1',
        'MSFT_log_return_lag3',
        'MSFT_log_return_lag5',
        'MSFT_log_return_lag10',
        'MSFT_log_return_lag15',
        'MSFT_log_return_lag21',
        'MSFT_rolling_vol_5',
        'MSFT_rolling_vol_21',
        'MSFT_log_volume',
        'SPY_log_return',
        'SPY_rolling_vol_5',
        'SPY_rolling_vol_21',
        'VIX_level',
        'VIX_log_change',
        'VIX_rolling_mean_21',
        'VXN_level',
        'VXN_log_change',
        'VXN_rolling_mean_21',
        'XLK_log_return',
        'MSFT_minus_XLK_return',
        'MSFT_minus_SPY_return',
        'QQQ_minus_SPY_return'
    ]

    additional_feature_cols = [
        'MSFT_rolling_vol_63',
        'MSFT_rolling_vol_126',
        'TNX_level',
        'TNX_change',
        'SOXX_log_return',
        'SOXX_rolling_vol_21',
        'TLT_log_return',
        'TLT_rolling_vol_21',
        'MSFT_log_volume_ratio',
        'MSFT_vol_5_lag1',
        'MSFT_vol_21_lag1',
        'MSFT_vol_63_lag1',
        'vol_ratio_5_21',
        'vol_ratio_21_63',
        'VIX_relative',
        'earnings_window'
    ]

    feature_cols = core_feature_cols + additional_feature_cols

    target_cols = [
        'target_volatility_1d',
        'target_volatility_5d',
        'target_volatility_21d'
    ]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df, feature_cols, target_cols
