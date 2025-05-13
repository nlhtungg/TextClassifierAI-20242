from flask import Flask, render_template, request, jsonify
from predict import predict_category, initialize_models
import time
from setup_models_util import setup_models_on_server

# Ensure models are downloaded and ready
setup_models_on_server()

# Initialize/load models into memory
initialize_models()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data.get('text', '')
    if input_text.strip():
        start_time = time.time()
        # predict_category will now use globally loaded models
        category = predict_category(input_text)
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        return jsonify({'category': category, 'processing_time': processing_time})
    return jsonify({'error': 'Input text is empty', 'processing_time': 0}), 400

if __name__ == '__main__':
    app.run(debug=True)
