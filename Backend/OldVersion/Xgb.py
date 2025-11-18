import numpy as np
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
libsvm_file = "data.libsvm"
X, y = load_svmlight_file(libsvm_file, n_features=13)  # Adjust n_features if needed

# Inspect data
print("Shape of X:", X.shape)
print("First few labels in y:", y[:5])
print("Unique labels in y:", np.unique(y))
print("Class distribution in y:", np.bincount(y.astype(int)))

# Check for binary classification
if len(np.unique(y)) < 2:
    raise ValueError("Dataset must contain at least two classes for binary classification.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost parameters
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",
    "scale_pos_weight": len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1,
}

# Train the model
num_rounds = 100
bst = xgb.train(params, dtrain, num_rounds)

# Predict and evaluate
y_pred = (bst.predict(dtest) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

y_train_pred = (bst.predict(dtrain) > 0.5).astype(int)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Visualize
plt.hist(bst.predict(dtest), bins=50)
plt.title("Predicted Probabilities")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()