import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("dataset/crop.csv")

X = data[['N','P','K','temperature','humidity','ph','rainfall']]
y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Crop Model Accuracy:", accuracy_score(y_test, pred))

# Save model
pickle.dump(model, open("models/crop_model.pkl","wb"))

print("Crop model saved successfully!")

