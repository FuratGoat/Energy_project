from data_fetcher import fetch_data
from data_processor import load_and_process_data
from feature_engineering import create_lagged_features
from model_training import train_and_evaluate_models, models
from future_prediction import predict_future
from plotter import plot_state_comparison, plot_predictions, plot_future_predictions, analyze_seasonality
import pandas as pd

# API Configuration
api_key = 'HR7xQ6gedHp2r1ixSmT75eAL84Zbffd8ipkE64D1'
url = f'https://api.eia.gov/v2/electricity/retail-sales/data/?api_key={api_key}&frequency=quarterly&data[0]=price&data[1]=revenue&start=2001-Q1&end=2024-Q1&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'

def main():
    # Fetch data
    data = fetch_data(api_key, url)
    
    # Load and process data
    df_state = load_and_process_data()
    if df_state is None:
        return
    
    # Plot state comparison
    states_to_plot = ['Hawaii', 'California', 'Rhode Island', 'Maine', 'Connecticut']
    plot_state_comparison(df_state, states_to_plot)
    
    # Focus on California
    california_prices = df_state['California'].dropna()
    
    # Analyze seasonality
    analyze_seasonality(california_prices)
    
    # Create lagged features for prediction
    features_df = create_lagged_features(california_prices)
    
    # Prepare data for modeling
    X = features_df.drop('price', axis=1)
    y = features_df['price']
    
    results, predictions, X_train, X_test, y_train, y_test = train_and_evaluate_models(X, y)
    
    # Number of future steps to predict (e.g., 4 quarters for 1 year ahead)
    n_future_steps = 4
    
    # Generate future dates based on the last date in the dataset
    last_date = california_prices.index[-2]
    future_dates = pd.date_range(last_date, periods=n_future_steps, freq='Q')
    
    future_predictions = {}
    
    for name, model in models.items():
        future_pred = predict_future(model, X, n_future_steps)
        future_predictions[name] = future_pred
    
    # Plot predictions and future forecasts
    plot_predictions(predictions, y_test, X_test)
    plot_future_predictions(predictions, future_predictions, y_test, future_dates)

if __name__ == "__main__":
    main()
