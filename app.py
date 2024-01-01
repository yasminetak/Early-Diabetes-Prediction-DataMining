from flask import Flask, render_template, request
import joblib
import numpy as np
import sklearn
print(sklearn.__version__)

app = Flask(__name__)

# Load your trained Random Forest model
model = joblib.load('random_forest_modelv1.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
        age = float(request.form['age'])

        # Add other variables...

        # Preprocess the input data
        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi,
                               diabetes_pedigree_function, age]).reshape(1, -1)

        # Ensure that the input_data has the correct number of features
        if input_data.shape[1] != 30:  # Update with the correct number of features
            return render_template('error.html', message="Number of features doesn't match.")

        # Make a prediction using the loaded model
        prediction = model.predict(input_data)[0]

        # Render the prediction on the result page
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
