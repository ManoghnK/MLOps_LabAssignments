import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from google.cloud import storage

# ----------------------------
# 1. Loading Dataset
# ----------------------------
data = load_iris()
X = data.data
y = data.target

# ----------------------------
# 2. Train / Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ----------------------------
# 3. Evaluate
# ----------------------------
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Model accuracy:", acc)

# ----------------------------
# 4. Save model locally
# ----------------------------
joblib.dump(model, "iris_model.pkl")
print("Model saved as iris_model.pkl")

# ----------------------------
# 5. Upload to GCS bucket
# ----------------------------
BUCKET_NAME = "terraform-lab-bucket-478605"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob("iris_model.pkl")
blob.upload_from_filename("iris_model.pkl")

print("Model uploaded to GCS bucket successfully!")
