import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# Định nghĩa các cột đặc trưng
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Hàm nhập dữ liệu từ bàn phím
def input_patient_data():
    print("Nhập thông tin bệnh nhân:")
    patient_data = {}
    
    # Mô tả và nhập dữ liệu cho từng đặc trưng
    print("\n1. Tuổi (age): số tuổi của bệnh nhân (ví dụ: 45)")
    patient_data["age"] = float(input("Nhập tuổi: "))
    
    print("\n2. Giới tính (sex): 1 = Nam, 0 = Nữ")
    patient_data["sex"] = float(input("Nhập giới tính (1 hoặc 0): "))
    
    print("\n3. Loại đau ngực (cp):")
    print("   0: Không đau ngực")
    print("   1: Đau thắt ngực điển hình")
    print("   2: Đau thắt ngực không điển hình")
    print("   3: Đau không do tim")
    patient_data["cp"] = float(input("Nhập loại đau ngực (0-3): "))
    
    print("\n4. Huyết áp nghỉ (trestbps): huyết áp tâm thu (mm Hg, ví dụ: 120)")
    patient_data["trestbps"] = float(input("Nhập huyết áp nghỉ: "))
    
    print("\n5. Cholesterol huyết thanh (chol): mg/dl (ví dụ: 200)")
    patient_data["chol"] = float(input("Nhập cholesterol: "))
    
    print("\n6. Đường huyết lúc đói (fbs): 1 nếu > 120 mg/dl, 0 nếu không")
    patient_data["fbs"] = float(input("Nhập đường huyết lúc đói (1 hoặc 0): "))
    
    print("\n7. Kết quả điện tâm đồ nghỉ (restecg):")
    print("   0: Bình thường")
    print("   1: Có bất thường sóng ST-T")
    print("   2: Phì đại thất trái")
    patient_data["restecg"] = float(input("Nhập kết quả điện tâm đồ (0-2): "))
    
    print("\n8. Nhịp tim tối đa (thalach): nhịp tim cao nhất đạt được (ví dụ: 150)")
    patient_data["thalach"] = float(input("Nhập nhịp tim tối đa: "))
    
    print("\n9. Đau thắt ngực do gắng sức (exang): 1 = Có, 0 = Không")
    patient_data["exang"] = float(input("Nhập đau thắt ngực do gắng sức (1 hoặc 0): "))
    
    print("\n10. Độ dốc ST (oldpeak): độ chênh lệch ST do gắng sức (ví dụ: 1.0)")
    patient_data["oldpeak"] = float(input("Nhập độ dốc ST: "))
    
    print("\n11. Độ dốc của đoạn ST (slope):")
    print("   1: Dốc lên")
    print("   2: Bằng phẳng")
    print("   3: Dốc xuống")
    patient_data["slope"] = float(input("Nhập độ dốc đoạn ST (1-3): "))
    
    print("\n12. Số mạch máu chính bị tắc (ca): 0-3")
    patient_data["ca"] = float(input("Nhập số mạch máu chính bị tắc (0-3): "))
    
    print("\n13. Kết quả kiểm tra thalassemia (thal):")
    print("   3: Bình thường")
    print("   6: Khiếm khuyết cố định")
    print("   7: Khiếm khuyết có thể đảo ngược")
    patient_data["thal"] = float(input("Nhập kết quả thalassemia (3, 6, hoặc 7): "))
    
    return patient_data

# Tải mô hình và scaler
try:
    model = tf.keras.models.load_model('heart_model_ann.keras')
    scaler = joblib.load('scaler.pkl')
    print("Đã tải mô hình và scaler thành công.")
except Exception as e:
    print(f"Lỗi khi tải mô hình hoặc scaler: {e}")
    exit()

# Nhập dữ liệu bệnh nhân
patient_data = input_patient_data()

# Chuyển dữ liệu thành DataFrame
patient_df = pd.DataFrame([patient_data], columns=columns)

# Chuẩn hóa dữ liệu
patient_scaled = scaler.transform(patient_df)

# Dự đoán
prediction_prob = model.predict(patient_scaled)
prediction = (prediction_prob > 0.5).astype(int)

# In kết quả
print("\nKết quả dự đoán:")
if prediction[0] == 1:
    print("Bệnh nhân có nguy cơ mắc bệnh tim.")
else:
    print("Bệnh nhân không có nguy cơ mắc bệnh tim.")
print(f"Xác suất nguy cơ bệnh tim: {prediction_prob[0][0]:.2%}")