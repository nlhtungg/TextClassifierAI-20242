from flask import Flask, render_template, request, jsonify, send_from_directory
import predict_model
import os

app = Flask(__name__, template_folder='.')
# Load model ngay khi khởi động

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    result = predict_model.predict(text)
    print(f"Received text: {text}")
    print(f"Prediction result: {result}")
    return jsonify({"label": result})

if __name__ == "__main__":
    app.run(debug=True)
