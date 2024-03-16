from flask import Flask, request, abort
import subprocess

from model import My_Classifier_Model

app = Flask(__name__)
model = My_Classifier_Model()

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        dataset_path = data['dataset_path']
        model.train(dataset_path)
        return "Model trained successfully!"
    except ValueError as e:
        return abort(500, str(e))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        dataset_path = data['dataset_path']
        model.predict(dataset_path)
        return "Model predicted successfully!"
    except ValueError as e:
        return abort(500, str(e))

if __name__ == '__main__':
  app.run(debug=True)