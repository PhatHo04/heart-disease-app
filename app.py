from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load mô hình và scaler
model = tf.keras.models.load_model('heart_model_ann.keras')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Lấy dữ liệu từ form
            features = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['restecg']),
                float(request.form['thalach']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal'])
            ]
            features = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features)

            # Dự đoán
            result = model.predict(features_scaled)[0][0]
            prediction = "Nguy cơ bệnh tim cao" if result > 0.5 else "Nguy cơ bệnh tim thấp"

        except Exception as e:
            prediction = f"Lỗi xử lý dữ liệu: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
