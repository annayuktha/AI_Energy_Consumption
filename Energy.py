import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset with error handling
try:
    df = pd.read_csv("C:/Yuktha/Tutor/E-Diploma/Industry_Oriented_Projects/Energy_Consumption/archive/KwhConsumptionBlower78_1.csv")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Combine 'TxnDate' and 'TxnTime' into a single datetime column
df["TxnDateTxnTime"] = pd.to_datetime(df["TxnDate"] + " " + df["TxnTime"], errors='coerce')

# Drop any rows with invalid datetime values
df.dropna(subset=["TxnDateTxnTime"], inplace=True)

# Set the new Datetime column as index
df.set_index("TxnDateTxnTime", inplace=True)

# Drop original 'TxnDate' and 'TxnTime' columns
df.drop(columns=["TxnDate", "TxnTime"], inplace=True)

# Ensure index uniqueness before concatenation
df = df[~df.index.duplicated(keep='first')]

# Extract hour and one-hot encode each day of the week
df["Hour"] = df.index.hour
day_of_week = pd.get_dummies(df.index.dayofweek, prefix="Day")
df = pd.concat([df, day_of_week], axis=1)

# Handle missing values
print("Before handling NaNs:", df.isna().sum())
df.fillna(df.mean(), inplace=True)
print("After handling NaNs:", df.isna().sum())

# Define input features and target variable
if "Consumption" not in df.columns:
    print("Error: 'Consumption' column not found in the dataset.")
    exit()
X = df.drop(columns=["Consumption"])
y = df["Consumption"]

# Ensure there is data before splitting
if X.empty or y.empty:
    print("Error: No data available after preprocessing.")
    exit()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Ensure data is not empty after split
if X_train.empty or X_test.empty:
    print("Error: Training or testing set is empty.")
    exit()

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check for NaNs after scaling
print("Checking for NaNs in X_train:", np.isnan(X_train_scaled).sum())
print("Checking for NaNs in X_test:", np.isnan(X_test_scaled).sum())

# Train an improved MLP Regressor
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),  # More complex architecture
    activation='relu',
    solver='adam',
    max_iter=1000,  # More iterations for better convergence
    early_stopping=True,  # Stops training if validation loss stops improving
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# Make predictions
y_pred = mlp.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', linestyle='dashed', alpha=0.7)
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Energy Consumption (MW)")
plt.title("Actual vs Predicted Energy Consumption")
plt.legend()
plt.grid(True)
plt.show()
