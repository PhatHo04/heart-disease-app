import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib

# Đặt seed để đảm bảo kết quả có thể tái hiện
np.random.seed(42)
tf.random.set_seed(42)

# Định nghĩa tên các cột
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Danh sách các file dữ liệu
files = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data"
]

# Đọc và kết hợp các tập dữ liệu
dataframes = []
for file in files:
    try:
        df = pd.read_csv(file, names=columns, na_values="?")
        dataframes.append(df)
        print(f"Đã đọc: {file}, số dòng: {len(df)}")
    except Exception as e:
        print(f"Lỗi khi đọc {file}: {e}")

df_all = pd.concat(dataframes, ignore_index=True)
print(f"Tổng số dòng trước khi xóa thiếu dữ liệu: {len(df_all)}")

# Làm sạch dữ liệu
df_all.dropna(inplace=True)
df_all = df_all.astype(float)
df_all["target"] = df_all["target"].apply(lambda x: 1 if x > 0 else 0)
print(f"Số dòng sau khi làm sạch: {len(df_all)}")

# Tách đặc trưng và nhãn
X = df_all.drop("target", axis=1)
y = df_all["target"]

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Mô hình ANN
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Dự đoán
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Lưu mô hình và scaler
model.save('heart_model_ann.keras')
joblib.dump(scaler, 'scaler.pkl')

# Đánh giá
print("\nĐộ chính xác:", accuracy_score(y_test, y_pred))
print("\nBáo cáo phân loại:\n", classification_report(y_test, y_pred))

# ========== BIỂU ĐỒ ==========
# 1. Biểu đồ loss và accuracy
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss theo Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Độ chính xác theo Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")  # Lưu biểu đồ
plt.show()

# 2. Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Âm tính', 'Dương tính'], yticklabels=['Âm tính', 'Dương tính'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')
plt.savefig("confusion_matrix.png")
plt.show()

# 3. ROC và AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Biểu đồ ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.savefig("roc_curve.png")
plt.show()
