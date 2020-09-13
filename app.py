import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
lr_model = pickle.load(open('lr_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = lr_model.predict(final_features)

    output = round(prediction[0], 2)
    result = output*100;

    return render_template('Home.html', prediction_text='Chances are {}%'.format(result))

"""
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
"""

if __name__ == "__main__":
    app.run(debug=True)
