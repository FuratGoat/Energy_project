import pandas as pd
import numpy as np

def create_lagged_features(df_prices, n_lags=4):
    """Create lagged features to predict future values"""
    df = pd.DataFrame(index=df_prices.index)
    
    # Basic time features
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    # Cyclical encoding of time features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lagged prices (past n quarters)
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df_prices.shift(lag)
    
    # Price momentum and changes
    df['price_change_1'] = df_prices.pct_change(periods=1).fillna(0)
    
    # Add the price itself as a feature (for training purposes)
    df['price'] = df_prices
    
    return df.fillna(method='ffill').fillna(method='bfill')
