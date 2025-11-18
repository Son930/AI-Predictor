import pandas as pd
import numpy as np
import xgboost as xgb
from colorama import deinit
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import time
import warnings

from torch.cuda import device

warnings.filterwarnings('ignore')

# Start timer
start_time = time.time()

# Load dataset
print("Loading and preparing data...")
data_path = "xau_usd_engineered.csv"
df = pd.read_csv(data_path)

# Handle missing values more carefully
# Check for missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(f"Found {missing_values.sum()} missing values")
    # Forward fill for time series data instead of dropping
    df = df.fillna(method='ffill').fillna(method='bfill')

# Define feature columns (excluding non-numeric and target)
feature_cols = [
    'open', 'high', 'low', 'close', 'volume', 'SMA', 'LMA', 'RSI', 'MACD', 'Signal',
    'Volatility', 'ATR', 'Relative_Volume', 'Volume_Change', 'Daily_Range', 'Price_Change',
    'Day_of_Week', 'Month_of_Year', 'Denoised_Price', 'Wavelet_D1', 'Wavelet_D4', 'Price_t-1',
    'Price_t-7', 'Price_t-14', 'BB_Upper', 'BB_Lower', 'BB_Bandwidth', 'EMA_Short',
    'EMA_Long', 'ROC', 'Stoch_K', 'Stoch_D', 'Return_t-1', 'Return_t-7', 'Return_t-14'
]

target_col = "Target"

# Check for infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
if df.isnull().sum().sum() > 0:
    df = df.fillna(method='ffill').fillna(method='bfill')


# Function to check if GPU is available
def is_gpu_available():
    try:
        # Create a small test XGBoost model with GPU
        test_model = xgb.XGBRegressor(tree_method='hist')
        test_X = np.random.random((10, 10))
        test_y = np.random.random(10)
        test_model.fit(test_X, test_y)
        return True
    except Exception:
        return False


# Check GPU availability
gpu_available = is_gpu_available()
print(f"GPU available for XGBoost: {gpu_available}")


# Feature selection - calculate feature importance from a base model
def select_important_features(X, y, feature_names, threshold=0.01):
    # Train a quick model to get feature importance
    base_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda"
    )
    base_model.fit(X, y)

    # Get feature importance
    importance = base_model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    # Select features above threshold
    selected_features = features_df[features_df['Importance'] > threshold]['Feature'].tolist()
    print(f"Selected {len(selected_features)} important features out of {len(feature_names)}")

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(features_df['Feature'][:15], features_df['Importance'][:15])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()
    plt.savefig('feature_importance.png')

    return selected_features


# Split data - use TimeSeriesSplit instead of random split for time series data
X = df[feature_cols]
y = df[target_col]

# Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

# Select important features
selected_features = select_important_features(X_scaled_df, y, feature_cols)
X_selected = X_scaled_df[selected_features]

# Use proper time series split for financial data
# Last 20% as test set, maintaining time order
split_idx = int(len(df) * 0.8)
X_train, X_test = X_selected.iloc[:split_idx], X_selected.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Hyperparameter tuning with cross-validation
print("Finding optimal hyperparameters with cross-validation...")
tscv = TimeSeriesSplit(n_splits=5)

# Define parameter grid - adjust based on available resources
param_grid = {
    'n_estimators': [300, 500],
    'learning_rate': [0.01, 0.05],
    'max_depth': [4, 6],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'gamma': [0, 0.1],
    'min_child_weight': [1, 3]
}

# Use early stopping to prevent overfitting
xgb_base = xgb.XGBRegressor(
    tree_method='hist',
    device="cuda",
    random_state=42
)

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=tscv,
    verbose=1,
    n_jobs=-1  # Use all CPU cores
)

grid_search.fit(X_train, y_train)

# Get best parameters and model
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train final model with best parameters
print("Training final model with optimal parameters...")
final_model = xgb.XGBRegressor(
    **best_params,
    tree_method='hist',
    device="cuda",
    random_state=42
)

# Add early stopping
eval_set = [(X_train, y_train), (X_test, y_test)]
final_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    eval_metric='rmse',
    early_stopping_rounds=50,
    verbose=True
)

# Evaluate model
y_pred = final_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.6f}")
print(f"Root Mean Squared Error: {rmse:.6f}")
print(f"Mean Absolute Error: {mae:.6f}")
print(f"R² Score: {r2:.6f}")

# Plot predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Gold Price Prediction - Actual vs Predicted')
plt.tight_layout()
plt.savefig('prediction_results.png')

# Save model and preprocessing objects
print("Saving model and preprocessing objects...")
joblib.dump(final_model, "xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_features, "selected_features.pkl")


# Create a minimal prediction function
def create_prediction_module():
    code = """
import pandas as pd
import numpy as np
import joblib

def predict_gold_price(new_data):
    \"\"\"
    Make predictions with the trained XGBoost model

    Args:
        new_data: DataFrame with the same features as training data

    Returns:
        Array of predictions
    \"\"\"
    # Load saved objects
    model = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    selected_features = joblib.load("selected_features.pkl")

    # Ensure all required features are present
    for feature in selected_features:
        if feature not in new_data.columns:
            raise ValueError(f"Missing required feature: {feature}")

    # Scale features
    X = new_data[selected_features]
    X_scaled = scaler.transform(X)

    # Make prediction
    predictions = model.predict(X_scaled)

    return predictions
"""

    with open("predict.py", "w") as f:
        f.write(code)


create_prediction_module()

# Print execution time
end_time = time.time()
print(f"\nExecution completed in {end_time - start_time:.2f} seconds")
print("Model saved as 'xgboost_model.pkl'")
print("Preprocessing objects saved as 'scaler.pkl' and 'selected_features.pkl'")
print("Prediction module saved as 'predict.py'")
print("Visualization saved as 'feature_importance.png' and 'prediction_results.png'")