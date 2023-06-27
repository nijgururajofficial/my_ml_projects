from flask import Flask, render_template, request
from joblib import load
import numpy as np
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')
    # return 'Hello, World!'

model = load('./savedmodels/model.joblib')

@app.route('/result',methods=['POST','GET'])
def predict():
    features = [int(x) for x in request.form.values()]
    final = [np.array(features)]
    result = model.predict(final)

    if result == 0:
        return render_template('result.html',result = "did not survive")
    else:
        return render_template('result.html',result = "survived!")

if __name__ == "__main__":
    app.run(debug=True, port=8000)