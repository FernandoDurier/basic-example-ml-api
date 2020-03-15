# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
app = Flask(__name__)# Load the model
model = pickle.load(open('model.pkl','rb'))

@app.route('/predict',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data['exp'])]])    # Take the first value of prediction
    output = prediction[0]    
    return jsonify(output)

@app.route('/fit',methods=['POST'])
def fit():
    data = request.get_json(force=True)
    X = data['features']
    y = data['labels']    
    model.fit(X,y)
    pickle.dump(regressor, open('model.pkl','wb'))# Loading model to compare the results
    model = pickle.load(open('model.pkl','rb'))
    return "Model trained."

if __name__ == '__main__':
    app.run(port=5000, debug=True)