import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("dataset/Fertilizer.csv")

# Use EXACT column names from your dataset
X = data[['Temparature', 'Humidity ', 'Moisture',
          'Nitrogen', 'Potassium', 'Phosphorous']]

y = data['Fertilizer Name']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Fertilizer Model Accuracy:", accuracy_score(y_test, pred))

# Save model
pickle.dump(model, open("models/fertilizer_model.pkl", "wb"))
print("Fertilizer model saved successfully!")

