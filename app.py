from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model, scaler, and polynomial features
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('poly.pkl', 'rb') as poly_file:
    poly = pickle.load(poly_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    stock = float(request.form['stock'])
    price = float(request.form['price'])
    total_price = float(request.form['total_price'])
    penjualan_m1 = float(request.form['penjualan_m1'])
    penjualan_m2 = float(request.form['penjualan_m2'])
    penjualan_m3 = float(request.form['penjualan_m3'])
    penjualan_m4 = float(request.form['penjualan_m4'])
    total_penjualan = float(request.form['total_penjualan'])
    rating = float(request.form['rating'])

    # Create feature array
    features = np.array([[stock, price, total_price, penjualan_m1, penjualan_m2, penjualan_m3, penjualan_m4, total_penjualan, rating]])

    # Apply polynomial transformation and scaling
    features_poly = poly.transform(features)
    features_poly_scaled = scaler.transform(features_poly)

    # Predict using the loaded model
    prediction = model.predict(features_poly_scaled)

    # Interpret prediction
    result = 'Iya' if prediction[0] == 1 else 'Tidak'

    return render_template('index.html', prediction_text=f'Prediksi Laku: {result}')



if __name__ == "__main__":
    app.run(debug=True)
