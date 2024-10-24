import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_state_comparison(df_state, states_to_plot):
    """Plot comparison of electricity prices across states"""
    plt.figure(figsize=(15, 8))
    
    for state in states_to_plot:
        if state in df_state.columns:
            plt.plot(df_state.index, df_state[state], label=state, linewidth=2)

    plt.title('Electricity Retail Sales in the Industrial Sector', fontsize=14, pad=20)
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Price (cents per kWh)', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_predictions(predictions, y_test, X_test):
    """Plot model predictions vs actual values"""
    plt.figure(figsize=(15, 8))
    
    plt.plot(y_test.index, y_test, 'b-', label='Actual Prices', alpha=0.5)
    
    for name, pred in predictions.items():
        plt.plot(X_test.index, pred, label=f'{name} Predictions')
    
    plt.title('Model Predictions vs Actual', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (cents per kWh)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_future_predictions(predictions, future_predictions, y_test, future_dates):
    """Plot model predictions vs actual values and future predictions"""
    plt.figure(figsize=(15, 8))
    
    # Plot actual test data
    plt.plot(y_test.index, y_test, 'b-', label='Actual Prices', alpha=0.5)
    
    # Plot model predictions on test set
    for name, pred in predictions.items():
        plt.plot(y_test.index, pred, label=f'{name} Predictions')
    
    # Plot future predictions
    for name, future_pred in future_predictions.items():
        plt.plot(future_dates, future_pred, '--', label=f'{name} Future Predictions')
    
    plt.title('Model Predictions and Future Forecasts', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (cents per kWh)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_seasonality(df_prices):
    """Analyze and plot seasonal decomposition"""
    decomposition = seasonal_decompose(df_prices, period=4)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
    
    decomposition.observed.plot(ax=ax1, linewidth=2)
    ax1.set_title('Observed', fontsize=12)
    
    decomposition.trend.plot(ax=ax2, linewidth=2, color='green')
    ax2.set_title('Trend', fontsize=12)
    
    decomposition.seasonal.plot(ax=ax3, linewidth=2, color='red')
    ax3.set_title('Seasonal', fontsize=12)
    
    decomposition.resid.plot(ax=ax4, linewidth=2, color='purple')
    ax4.set_title('Residual', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return decomposition
