"""
Time Series Forecasting Examples
================================

- Stock Price Prediction (LSTM, Prophet)
- Sales Forecasting (ARIMA, SARIMA)
- Energy Consumption Prediction
- Time series preprocessing and feature engineering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time series analysis libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine learning libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Prophet for time series forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")

# 1. Stock Price Prediction with LSTM
print("=== Stock Price Prediction with LSTM ===")

def generate_stock_data(n_days=500):
    """Generate synthetic stock price data"""
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate price data with trend and noise
    initial_price = 100
    prices = [initial_price]
    
    for i in range(1, n_days):
        # Add trend and random walk
        trend = 0.001  # Slight upward trend
        noise = np.random.normal(0, 0.02)
        price_change = trend + noise
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    # Add some volatility
    prices = np.array(prices)
    prices += np.random.normal(0, 0.5, len(prices))
    
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, n_days)
    })

def prepare_lstm_data(data, lookback=60):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']].values)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def create_lstm_model(lookback):
    """Create LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Generate and prepare stock data
stock_data = generate_stock_data()
print(f"Stock data shape: {stock_data.shape}")
print(f"Date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")

# Prepare data for LSTM
lookback = 60
X, y, scaler = prepare_lstm_data(stock_data, lookback)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create and train LSTM model
lstm_model = create_lstm_model(lookback)
print("Training LSTM model...")
history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

# Make predictions
train_predict = lstm_model.predict(X_train)
test_predict = lstm_model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict))
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# 2. Sales Forecasting with ARIMA/SARIMA
print("\n=== Sales Forecasting with ARIMA/SARIMA ===")

def generate_sales_data(n_periods=200):
    """Generate synthetic sales data with trend and seasonality"""
    np.random.seed(42)
    
    # Generate dates (monthly data)
    start_date = datetime(2018, 1, 1)
    dates = [start_date + timedelta(days=30*i) for i in range(n_periods)]
    
    # Generate sales with trend, seasonality, and noise
    t = np.arange(n_periods)
    trend = 100 + 2 * t  # Linear trend
    seasonality = 20 * np.sin(2 * np.pi * t / 12)  # Annual seasonality
    noise = np.random.normal(0, 10, n_periods)
    
    sales = trend + seasonality + noise
    sales = np.maximum(sales, 0)  # Sales can't be negative
    
    return pd.DataFrame({
        'Date': dates,
        'Sales': sales
    }).set_index('Date')

def check_stationarity(timeseries):
    """Check if time series is stationary"""
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    return result[1] < 0.05

def make_stationary(timeseries):
    """Make time series stationary by differencing"""
    diff_series = timeseries.diff().dropna()
    return diff_series

# Generate sales data
sales_data = generate_sales_data()
print(f"Sales data shape: {sales_data.shape}")

# Check stationarity
print("\nChecking stationarity of original series:")
is_stationary = check_stationarity(sales_data['Sales'])

if not is_stationary:
    print("\nSeries is not stationary. Making it stationary...")
    sales_diff = make_stationary(sales_data['Sales'])
    print("\nChecking stationarity of differenced series:")
    check_stationarity(sales_diff)
else:
    sales_diff = sales_data['Sales']

# Plot ACF and PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(sales_diff, ax=axes[0], lags=40)
plot_pacf(sales_diff, ax=axes[1], lags=40)
plt.tight_layout()
plt.show()

# Fit ARIMA model
print("\nFitting ARIMA model...")
# Based on ACF/PACF plots, try ARIMA(1,1,1)
arima_model = ARIMA(sales_data['Sales'], order=(1, 1, 1))
arima_fit = arima_model.fit()

print(f"ARIMA AIC: {arima_fit.aic:.2f}")
print(f"ARIMA BIC: {arima_fit.bic:.2f}")

# Fit SARIMA model for seasonal data
print("\nFitting SARIMA model...")
sarima_model = SARIMAX(sales_data['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)

print(f"SARIMA AIC: {sarima_fit.aic:.2f}")
print(f"SARIMA BIC: {sarima_fit.bic:.2f}")

# Make forecasts
forecast_steps = 12
arima_forecast = arima_fit.forecast(steps=forecast_steps)
sarima_forecast = sarima_fit.forecast(steps=forecast_steps)

print(f"\nARIMA forecast (next {forecast_steps} periods):")
print(arima_forecast)
print(f"\nSARIMA forecast (next {forecast_steps} periods):")
print(sarima_forecast)

# 3. Energy Consumption Prediction
print("\n=== Energy Consumption Prediction ===")

def generate_energy_data(n_days=365):
    """Generate synthetic energy consumption data"""
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate consumption with daily and weekly patterns
    t = np.arange(n_days)
    
    # Base consumption
    base_consumption = 1000
    
    # Daily pattern (higher during day, lower at night)
    daily_pattern = 200 * np.sin(2 * np.pi * t / 24)
    
    # Weekly pattern (lower on weekends)
    weekly_pattern = np.array([0 if i % 7 in [5, 6] else 50 for i in range(n_days)])
    
    # Seasonal pattern (higher in winter and summer)
    seasonal_pattern = 100 * np.sin(2 * np.pi * t / 365)
    
    # Random noise
    noise = np.random.normal(0, 30, n_days)
    
    consumption = base_consumption + daily_pattern + weekly_pattern + seasonal_pattern + noise
    consumption = np.maximum(consumption, 0)
    
    return pd.DataFrame({
        'Date': dates,
        'Consumption': consumption,
        'Temperature': 20 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 5, n_days)
    }).set_index('Date')

# Generate energy data
energy_data = generate_energy_data()
print(f"Energy data shape: {energy_data.shape}")

# Decompose time series
decomposition = seasonal_decompose(energy_data['Consumption'], model='additive', period=24)

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()

# Feature engineering for energy prediction
energy_data['Hour'] = energy_data.index.hour
energy_data['DayOfWeek'] = energy_data.index.dayofweek
energy_data['Month'] = energy_data.index.month
energy_data['DayOfYear'] = energy_data.index.dayofyear

# Create lag features
energy_data['Consumption_Lag1'] = energy_data['Consumption'].shift(1)
energy_data['Consumption_Lag24'] = energy_data['Consumption'].shift(24)
energy_data['Temperature_Lag1'] = energy_data['Temperature'].shift(1)

# Drop NaN values
energy_data = energy_data.dropna()

# Prepare features for ML model
features = ['Hour', 'DayOfWeek', 'Month', 'DayOfYear', 'Temperature', 
           'Consumption_Lag1', 'Consumption_Lag24', 'Temperature_Lag1']
X_energy = energy_data[features]
y_energy = energy_data['Consumption']

# Split data
train_size_energy = int(len(X_energy) * 0.8)
X_train_energy = X_energy[:train_size_energy]
X_test_energy = X_energy[train_size_energy:]
y_train_energy = y_energy[:train_size_energy]
y_test_energy = y_energy[train_size_energy:]

# Scale features
scaler_energy = MinMaxScaler()
X_train_scaled = scaler_energy.fit_transform(X_train_energy)
X_test_scaled = scaler_energy.transform(X_test_energy)

# Train simple ML model (Random Forest)
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train_energy)

# Make predictions
y_pred_energy = rf_model.predict(X_test_scaled)

# Calculate metrics
energy_rmse = np.sqrt(mean_squared_error(y_test_energy, y_pred_energy))
energy_mae = mean_absolute_error(y_test_energy, y_pred_energy)
print(f"Energy prediction RMSE: {energy_rmse:.2f}")
print(f"Energy prediction MAE: {energy_mae:.2f}")

# 4. Prophet Forecasting (if available)
if PROPHET_AVAILABLE:
    print("\n=== Prophet Forecasting ===")
    
    # Prepare data for Prophet
    prophet_data = stock_data.copy()
    prophet_data.columns = ['ds', 'y']  # Prophet requires 'ds' and 'y' columns
    
    # Split data
    train_size_prophet = int(len(prophet_data) * 0.8)
    train_prophet = prophet_data[:train_size_prophet]
    test_prophet = prophet_data[train_size_prophet:]
    
    # Create and fit Prophet model
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    prophet_model.fit(train_prophet)
    
    # Make forecast
    future = prophet_model.make_future_dataframe(periods=len(test_prophet))
    forecast = prophet_model.predict(future)
    
    # Calculate metrics
    prophet_predictions = forecast['yhat'][train_size_prophet:].values
    prophet_actual = test_prophet['y'].values
    prophet_rmse = np.sqrt(mean_squared_error(prophet_actual, prophet_predictions))
    print(f"Prophet RMSE: {prophet_rmse:.4f}")
    
    # Plot Prophet forecast
    fig = prophet_model.plot(forecast)
    plt.title('Prophet Forecast')
    plt.show()
    
    # Plot components
    fig = prophet_model.plot_components(forecast)
    plt.show()

# 5. Visualization and Comparison
print("\n=== Visualization and Comparison ===")

# Plot stock price predictions
plt.figure(figsize=(15, 10))

# Stock price LSTM results
plt.subplot(2, 3, 1)
plt.plot(stock_data['Date'][lookback:train_size+lookback], y_train_inv, label='Actual (Train)', alpha=0.7)
plt.plot(stock_data['Date'][lookback:train_size+lookback], train_predict, label='Predicted (Train)', alpha=0.7)
plt.plot(stock_data['Date'][train_size+lookback:], y_test_inv, label='Actual (Test)', alpha=0.7)
plt.plot(stock_data['Date'][train_size+lookback:], test_predict, label='Predicted (Test)', alpha=0.7)
plt.title('Stock Price Prediction (LSTM)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)

# Sales forecasting results
plt.subplot(2, 3, 2)
plt.plot(sales_data.index, sales_data['Sales'], label='Actual Sales', alpha=0.7)
plt.plot(arima_forecast.index, arima_forecast, label='ARIMA Forecast', alpha=0.7)
plt.plot(sarima_forecast.index, sarima_forecast, label='SARIMA Forecast', alpha=0.7)
plt.title('Sales Forecasting')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.xticks(rotation=45)

# Energy consumption results
plt.subplot(2, 3, 3)
test_dates = energy_data.index[train_size_energy:]
plt.plot(test_dates, y_test_energy, label='Actual', alpha=0.7)
plt.plot(test_dates, y_pred_energy, label='Predicted', alpha=0.7)
plt.title('Energy Consumption Prediction')
plt.xlabel('Date')
plt.ylabel('Consumption')
plt.legend()
plt.xticks(rotation=45)

# Model performance comparison
plt.subplot(2, 3, 4)
models = ['LSTM', 'ARIMA', 'SARIMA', 'Random Forest']
rmse_scores = [test_rmse, np.nan, np.nan, energy_rmse]  # ARIMA/SARIMA RMSE not calculated for brevity
plt.bar(models, rmse_scores, alpha=0.7)
plt.title('Model Performance Comparison (RMSE)')
plt.ylabel('RMSE')
plt.xticks(rotation=45)

# Time series decomposition
plt.subplot(2, 3, 5)
plt.plot(decomposition.seasonal[:100], label='Seasonal Pattern')
plt.title('Seasonal Decomposition (Energy)')
plt.xlabel('Time')
plt.ylabel('Seasonal Component')
plt.legend()

# Feature importance (for Random Forest)
plt.subplot(2, 3, 6)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance (Energy Prediction)')
plt.xlabel('Importance')

plt.tight_layout()
plt.show()

# 6. Summary and Recommendations
print("\n=== Summary and Recommendations ===")
print("Time Series Forecasting Results:")
print(f"1. Stock Price Prediction (LSTM):")
print(f"   - Train RMSE: {train_rmse:.4f}")
print(f"   - Test RMSE: {test_rmse:.4f}")
print(f"   - Best for: Complex patterns, long sequences")

print(f"\n2. Sales Forecasting (ARIMA/SARIMA):")
print(f"   - ARIMA AIC: {arima_fit.aic:.2f}")
print(f"   - SARIMA AIC: {sarima_fit.aic:.2f}")
print(f"   - Best for: Trend and seasonality")

print(f"\n3. Energy Consumption Prediction (Random Forest):")
print(f"   - RMSE: {energy_rmse:.2f}")
print(f"   - MAE: {energy_mae:.2f}")
print(f"   - Best for: Multiple features, non-linear patterns")

if PROPHET_AVAILABLE:
    print(f"\n4. Prophet Forecasting:")
    print(f"   - RMSE: {prophet_rmse:.4f}")
    print(f"   - Best for: Automatic seasonality, holidays, trend changes")

print(f"\nRecommendations:")
print(f"- Use LSTM for complex, long-term dependencies")
print(f"- Use ARIMA/SARIMA for univariate time series with clear patterns")
print(f"- Use Prophet for automatic seasonality and holiday effects")
print(f"- Use Random Forest for multivariate time series with feature engineering")
print(f"- Always check stationarity and decompose time series before modeling") 