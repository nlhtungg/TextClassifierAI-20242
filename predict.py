import joblib
import numpy as np
import re
import gensim
from sklearn.preprocessing import LabelEncoder

# Global variables for models, to be loaded by initialize_models()
svm_clf = None
w2v_model = None
label_encoder = None

# Danh sách stopwords, bạn có thể thay thế bằng danh sách stopwords thực tế của mình
stopwords = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"])

def initialize_models():
    """Loads all models from disk into global variables."""
    global svm_clf, w2v_model, label_encoder
    
    model_path_svm = 'model/svm_classifier_pipeline.pkl'
    model_path_w2v = 'model/word2vec_embedding.model'
    model_path_label_encoder = 'model/label_encoder.pkl'

    try:
        svm_clf = joblib.load(model_path_svm)
        w2v_model = gensim.models.Word2Vec.load(model_path_w2v)
        label_encoder = joblib.load(model_path_label_encoder)
        print("Models loaded successfully.") # Optional: for confirmation
    except FileNotFoundError as e:
        print(f"Error loading models: {e}. Ensure models are downloaded and paths are correct.")
        # Depending on desired behavior, you might want to raise e or exit
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading models: {e}")
        raise


# Hàm dự đoán nhãn cho văn bản
def predict_category(text, vector_size=300): # w2v_model and clf removed from params
    global svm_clf, w2v_model, label_encoder # Ensure we are using the global models

    if not all([svm_clf, w2v_model, label_encoder]):
        # This check is a fallback, initialize_models should be called first.
        print("Models not loaded. Call initialize_models() first.")
        # Attempt to load them now, or handle error appropriately
        initialize_models() 
        # If initialize_models() fails, it will raise an exception.
        # If it succeeds, the global variables will be populated.

    # Làm sạch văn bản
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Bỏ dấu câu
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords]

    # Vector hóa bằng Word2Vec (trung bình vector các từ)
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if len(vectors) == 0:
        vector = np.zeros(vector_size).reshape(1, -1)
    else:
        vector = np.mean(vectors, axis=0).reshape(1, -1)

    # Dự đoán nhãn
    label_id = svm_clf.predict(vector)[0] # Use global svm_clf
    label_name = label_encoder.inverse_transform([label_id])[0] # Use global label_encoder

    return label_name

if __name__ == "__main__":
    # Initialize models when running script directly for testing
    try:
        initialize_models()
        # Kiểm tra dự đoán
        test_sentence = "A bird flies on the sky"

        # Pass only the text, as w2v_model is now accessed globally
        predicted_category = predict_category(test_sentence) 
        print(f"📝 Câu: {test_sentence}")
        print(f"📂 Dự đoán thuộc nhãn: {predicted_category}")
    except Exception as e:
        print(f"Error in main execution: {e}")
