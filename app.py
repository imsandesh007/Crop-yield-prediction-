from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model and preprocessor
ls = pickle.load(open('ls.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            Taluka_Name = request.form['taluka_name']
            Crop_Varieties = request.form['crop_varieties']
            N_kg_ha = float(request.form['n_kg_ha'])
            P_kg_ha = float(request.form['p_kg_ha'])
            K_kg_ha = float(request.form['k_kg_ha'])
            pH = float(request.form['pH'])
            Rainfall_mm = int(request.form['rainfall'])
            Temperature_C = int(request.form['temperature'])

            # Prepare features for prediction
            features = np.array([[Taluka_Name, Crop_Varieties, N_kg_ha, P_kg_ha, K_kg_ha, pH, Rainfall_mm, Temperature_C]])
            transformed_features = preprocessor.transform(features)
            predicted_value = ls.predict(transformed_features).reshape(1, -1)

            # Render template with prediction
            return render_template('index.html', predicted_value=predicted_value[0][0])
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)