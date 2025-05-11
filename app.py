from setup_models_util import setup_models_on_server # Or your actual filename

# Attempt to download and set up models when the app starts
# This needs to happen BEFORE predict.py (and its model loading) is imported
if not setup_models_on_server():
    # You might want to raise an error or prevent the app from starting
    print("CRITICAL: Failed to setup models. Application might not function correctly.")
    # For a production app, you might sys.exit(1) or use a more robust error handling

from flask import Flask, render_template, request, jsonify
from predict import predict_category, w2v_model  
import time

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
        # Assuming predict_category and w2v_model are correctly imported and loaded
        category = predict_category(input_text, w2v_model)
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        return jsonify({'category': category, 'processing_time': processing_time})
    return jsonify({'error': 'Input text is empty', 'processing_time': 0}), 400

if __name__ == '__main__':
    app.run(debug=True)
