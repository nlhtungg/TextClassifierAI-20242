import joblib
import numpy as np
import re
import gensim
from sklearn.preprocessing import LabelEncoder

# Tải mô hình SVM và Word2Vec
svm_clf = joblib.load('model/svm_classifier_pipeline.pkl.gz')  # Load the gzipped SVM model
w2v_model = gensim.models.Word2Vec.load('model/word2vec_embedding.model')  # Tải mô hình Word2Vec
label_encoder = joblib.load('model/label_encoder.pkl')  # Tải LabelEncoder

# Danh sách stopwords, bạn có thể thay thế bằng danh sách stopwords thực tế của mình
stopwords = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"])


# Hàm dự đoán nhãn cho văn bản
def predict_category(text, w2v_model, vector_size=300, clf=svm_clf):
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
    label_id = clf.predict(vector)[0]
    label_name = label_encoder.inverse_transform([label_id])[0]

    return label_name
if __name__ == "__main__":
    # Kiểm tra dự đoán
    test_sentence = "A bird flies on the sky"

    predicted_category = predict_category(test_sentence, w2v_model)
    print(f"📝 Câu: {test_sentence}")
    print(f"📂 Dự đoán thuộc nhãn: {predicted_category}")
