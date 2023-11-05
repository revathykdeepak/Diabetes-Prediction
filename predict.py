import pickle
import xgboost as xgb
from flask import Flask, request, jsonify

app = Flask('diabetes_prediction')

model_file = 'xgb_trained_model.bin'

with open(model_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)  

def predict_single(person, dv, model):
    X = dv.transform([person])
    features = dv.get_feature_names_out()
    dpred = xgb.DMatrix(X, feature_names=list(features))
    y_pred = model.predict(dpred)
    return y_pred[0]

@app.route('/predict', methods=['POST']) 
def predict():
    person = request.get_json()  

    prediction = predict_single(person, dv, model)
    diabetes = prediction >= 0.5

    result = {
        'diabetes_probability': float(prediction), 
        'has_diabetis': bool(diabetes),  
    }

    return jsonify(result)  

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696)