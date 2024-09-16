from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
data = pd.read_csv('E:\\assignment_flask\\winequality-red (1).csv')
with open('model.pkl','rb') as model_file:
    rf_model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('wine.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    data = [float(request.form['fixed_acidity']),
            float(request.form['volatile_acidity']),
            float(request.form['citric_acid']),
            float(request.form['residual_sugar']),
            float(request.form['chlorides']),
            float(request.form['free_sulfur_dioxide']),
            float(request.form['total_sulfur_dioxide']),
            float(request.form['density']),
            float(request.form['pH']),
            float(request.form['sulphates']),
            float(request.form['alcohol'])]

    # Convert data to numpy array
    data = np.array([data])

    # Make prediction
    prediction = rf_model.predict(data)[0]

    return f"<h1>The predicted wine quality is: {prediction}</h1>"

if __name__ == '__main__':
    app.run(debug=True)
