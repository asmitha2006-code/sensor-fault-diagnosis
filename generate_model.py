import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Generate sensor data
normal_temp = np.random.uniform(20, 80, 800)
fault_temp = np.random.uniform(90, 120, 200)
temperature = np.concatenate([normal_temp, fault_temp])

normal_volt = np.random.uniform(3, 5.5, 800)
fault_volt1 = np.random.uniform(6, 7, 100)
fault_volt2 = np.random.uniform(1, 2, 100)
voltage = np.concatenate([normal_volt, fault_volt1, fault_volt2])

normal_current = np.random.uniform(0.1, 2, 800)
fault_current = np.random.uniform(0.01, 0.05, 200)
current = np.concatenate([normal_current, fault_current])

fault = np.concatenate([np.zeros(800), np.ones(200)])

df = pd.DataFrame({
    'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='1min'),
    'temperature': temperature,
    'voltage': voltage,
    'current': current,
    'fault': fault
})

X = df[['temperature', 'voltage', 'current']]
y = df['fault']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
joblib.dump(model, 'fault_model.pkl')
df.to_csv('sensor_data.csv', index=False)
print("Model and dataset saved.")
