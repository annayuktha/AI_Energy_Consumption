import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("C:/Yuktha/Tutor/E-Diploma/Industry_Oriented_Projects/Energy_Consumption/archive/KwhConsumptionBlower78_1.csv")

# Combine 'TxnDate' and 'TxnTime' into a single datetime column
df["TxnDateTxnTime"] = pd.to_datetime(df["TxnDate"] + " " + df["TxnTime"], errors='coerce')

# Drop rows with invalid datetime values
df.dropna(subset=["TxnDateTxnTime"], inplace=True)

# Set Datetime index and drop original date/time columns
df.set_index("TxnDateTxnTime", inplace=True)
df.drop(columns=["TxnDate", "TxnTime"], inplace=True)

# Remove duplicate indices
df = df[~df.index.duplicated(keep='first')]

# Extract hour and one-hot encode each day of the week
df["Hour"] = df.index.hour
day_of_week = pd.get_dummies(df.index.dayofweek, prefix="Day")

# Fill missing values in one-hot encoded days
day_of_week.fillna(0, inplace=True)

# Merge with original dataframe
df = pd.concat([df, day_of_week], axis=1)

# Fill NaN in 'Consumption' with median
df["Consumption"] = df["Consumption"].fillna(df["Consumption"].median())

# Define input features and target variable
X = df.drop(columns=["Consumption"])
y = df["Consumption"]

# Ensure no missing values
X.fillna(X.mean(), inplace=True)

# Check for remaining NaN values (debugging)
print("Missing values after handling:\n", X.isnull().sum())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MLP Regressor
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    early_stopping=True,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(mlp, "mlp_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")

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
