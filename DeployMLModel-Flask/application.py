import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


application = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict():
   
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('C:\\Users\\User\\Downloads\\DeployMLModel-Flask-20241203T085922Z-001\\DeployMLModel-Flask\\templates\\index.html', prediction_text='Employee Salary should be $ {}'.format(output))

if __name__ == "__main__":
    application.run(debug=True)
