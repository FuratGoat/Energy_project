import pandas as pd

def predict_future(model, X, n_steps):
    """
    Predict future values by iteratively using previous predictions.
    :param model: The trained model to use for prediction
    :param X: The most recent feature set to base future predictions on
    :param n_steps: The number of future time steps to predict
    :return: List of future predictions
    """
    future_predictions = []
    last_X = X.iloc[-1].copy()  # Start with the last known data point

    for _ in range(n_steps):
        # Predict the next time step
        next_prediction = model.predict(last_X.values.reshape(1, -1))[0]
        
        # Append the prediction to the list
        future_predictions.append(next_prediction)
        
        # Shift lagged features: Move all lagged features back by 1 step
        for lag in range(1, 4):  # Assuming you're using lag_1, lag_2, lag_3
            if f'lag_{lag+1}' in last_X:
                last_X[f'lag_{lag}'] = last_X[f'lag_{lag+1}']
        
        # Update the most recent lag_1 with the new prediction
        last_X[f'lag_1'] = next_prediction
    
    return future_predictions
