import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Đọc dữ liệu từ file CSV
data = pd.read_csv("hand_data.csv", header=None)

# Cột đầu tiên là label
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tạo model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Tính độ chính xác
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Lưu model
joblib.dump(model, "hand_model.pkl")
print("Model saved as hand_model.pkl")