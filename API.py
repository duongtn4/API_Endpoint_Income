from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import os
import sys


app = Flask(__name__)
@app.route('/incomepredict', methods = ['POST'])
def predict():
    if LR_Model:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            scaler.fit(query)
            scaler.transform(query)
            query = query.reindex(columns = model_columns, fill_value = 0)
            
            prediction = list(LR_Model.predict(query))
            
            return jsonify({'Result': {"Income Prediction":str(prediction)}})
        
        except:
            
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 5000
    os.chdir(r"C:\Users\thuyd\OneDrive\Documents\Sally\Take_Home_Assignment")
    LR_Model = joblib.load("model.pkl")
    print('Model loaded')
    model_columns = joblib.load('model_columns.pkl')
    print('Model columns loaded')
    scaler = joblib.load("Normalize.pkl")
    print('Normalization loaded')
    
    app.run(port=port, debug = True)
