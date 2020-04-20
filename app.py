import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
from statsmodels.iolib.smpickle import load_pickle
model = load_pickle("slr_wcat.pkl")

@app.route('/')
def home():
    return render_template('startup.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    waist = float_features[0];
    waist_sq = waist*waist
    waist_cb = waist*waist*waist
    wcat = pd.DataFrame([[waist, waist_sq, waist_cb]], columns=["Waist", "Waist_sq", "Waist_cb"])
    x = np.exp(model.predict(wcat))
    print(float(round(x,2)))
#    flt_features = [float(x) for x in request.form.values()]
##    int_features = [int(x) for x in request.form.values()]
#    final_features = [np.array(int_features)]
#    prediction = model.predict(final_features)
#
#    output = round(prediction[0], 2)
#
#    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    return render_template('startup.html', prediction_text= "Adipose Tissue Size is {}".format(float(round(x,2))))


if __name__ == "__main__":
    app.run(debug=True)