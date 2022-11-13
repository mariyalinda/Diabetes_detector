import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
from sklearn.preprocessing import StandardScaler
import pickle
app=Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
dataset=pd.read_csv('diabetes.csv')
X=dataset.drop(columns='Outcome',axis=1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    scaler=StandardScaler()
    scaler.fit(X)
    float_features = [float(x) for x in request.form.values()]
    nparray_features = np.asarray(float_features)
    input_data_reshaped=nparray_features.reshape(1,-1)
    std_data=scaler.transform(input_data_reshaped)
    prediction = model.predict(std_data) #array containing one element-result

    output = prediction[0]
    if(output==0):
        return render_template('index.html', prediction_text="Diabetes is not detected")
    elif(output==1):
        return render_template('index.html', prediction_text="Diabetes is detected")


if __name__ == "__main__":
    app.run(debug=True)