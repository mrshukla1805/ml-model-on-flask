import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	l=[]
	l.append(request.form['value1'])
	l.append(request.form['value2'])
	l.append(request.form['value3'])
	l.append(request.form['value4'])
	l_final = [np.array(l)]
	prediction =model.predict(l_final)

	return render_template('index.html',prediction=prediction)



if __name__ == "__main__":
    app.run(debug=True)