from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Global models dictionary
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

def train_and_evaluate_models(X, y):
    """Train and evaluate models with a fixed train-test split for small datasets"""
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)  # 30% test set
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"\n{name} Results:")
        
        model.fit(X_train, y_train)
        
        # Generate predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate performance metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        results[name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        predictions[name] = test_pred
        
        print(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        print(f"Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
    
    return results, predictions, X_train, X_test, y_train, y_test
