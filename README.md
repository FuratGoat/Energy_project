Welcome to the Electricity Price Forecasting project! 

This project is designed to analyze and predict electricity retail prices in the industrial sector using data from the U.S. Energy Information Administration (EIA).

Project Overview:
The project focuses on five U.S. states with the highest retail electricity prices. 
We use several machine learning models—Linear Regression, Random Forest, and Gradient Boosting—to predict future electricity prices. 
In addition, we analyze trends, seasonality, and residuals using seasonal decomposition.

How It Works:
The project pulls electricity price data from the EIA API.
It processes the data, adds lagged features, and uses machine learning models to predict prices.
Seasonal decomposition is performed to show trends, seasonal patterns, and residuals.
Future predictions for electricity prices are made for the next four quarters.
Project Structure

Here's a breakdown of the key files:
data_fetcher.py: Fetches electricity price data from the EIA API.
data_processor.py: Processes the data into a usable format.
feature_engineering.py: Creates lagged features and other useful indicators for model training.
model_training.py: Trains machine learning models and evaluates their performance.
future_prediction.py: Handles future price predictions using trained models.
plotter.py: Contains plotting functions for visualizing trends, model predictions, and future forecasts.
main.py: Orchestrates the entire process by calling the necessary functions from other files.

You’ll see:
Comparison of electricity prices for the 5 states with the highest yearly average change.
Seasonal decomposition for California electricity prices.
Predictions for future electricity prices for the next four quarters.

Future Work:
Extend the models to include more states or regions.
Experiment with other machine learning algorithms for better accuracy.
Explore additional features that might affect electricity prices (e.g., weather, economic indicators)
