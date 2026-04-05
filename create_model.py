import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Dummy data for demonstration
np.random.seed(42)
X = np.random.rand(100, 6)
y = np.random.randint(0, 2, 100)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "pcos_model.pkl")
print("Model saved as pcos_model.pkl")