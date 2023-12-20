import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score

dataset_path = 'archive/data_2m.csv'  
data = pd.read_csv(dataset_path)
X = data.drop('type', axis=1)
y = data['type']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_val_pred = rf_model.predict(X_val)

mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"Mean Squared Error (MSE) on validation set: {mse}")
print(f"R-squared (R2) on validation set: {r2}")

y_test_pred = rf_model.predict(X_test)

mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"\nMean Squared Error (MSE) on test set: {mse_test}")
print(f"R-squared (R2) on test set: {r2_test}")

