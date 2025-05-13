import math
import random
from collections import Counter, defaultdict
import time
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV

# Giữ nguyên dữ liệu X_train đã được cắt mẫu
X_train = X_train
y_train = y_train
X_test = X_test
y_test = y_test

# Khởi tạo các mô hình cơ sở
models = [LogisticRegression(), GaussianNB(), LinearSVM(), RandomForest()]

# Tạo mô hình ensemble với Voting Classifier (bỏ GradientBoostingClassifier)
ensemble = VotingClassifier(estimators=[
    ('lr', models[0]), 
    ('nb', models[1]), 
    ('svm', models[2]), 
    ('rf', models[3])], 
    voting='soft')

# Huấn luyện mô hình ensemble trực tiếp mà không dùng GridSearchCV
start_time = time.time()
ensemble.fit(X_train, y_train)
end_time = time.time()
print(f"⏱️ Thời gian huấn luyện (không GridSearchCV): {end_time - start_time:.2f} giây")

# Đánh giá trên tập train
y_train_pred = ensemble.predict(X_train)
y_train_proba = ensemble.predict_proba(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_loss = log_loss(y_train, y_train_proba)
print(f"📈 Train Accuracy: {train_accuracy:.2f}")
print(f"📉 Train Log Loss : {train_loss:.4f}")

# Đánh giá trên tập test
y_pred = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_loss = log_loss(y_test, y_proba)
print(f"Test Accuracy : {test_accuracy:.2f}")
print(f"Test Log Loss : {test_loss:.4f}")

# Báo cáo chi tiết
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Huấn luyện với GridSearchCV
param_grid = {
    'voting': ['hard', 'soft'],
    'weights': [[1, 1, 1, 1], [2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]]
}

grid_search = GridSearchCV(ensemble, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"Thời gian huấn luyện (GridSearchCV): {end_time - start_time:.2f} giây")
print(f"Best Params: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.2f}")

# Đánh giá trên tập test sau khi tối ưu
best_ensemble = grid_search.best_estimator_
y_pred_gs = best_ensemble.predict(X_test)
y_proba_gs = best_ensemble.predict_proba(X_test)

test_accuracy_gs = accuracy_score(y_test, y_pred_gs)
test_loss_gs = log_loss(y_test, y_proba_gs)
print(f"Test Accuracy (GridSearchCV) : {test_accuracy_gs:.2f}")
print(f"Test Log Loss (GridSearchCV) : {test_loss_gs:.4f}")

# Báo cáo chi tiết sau tối ưu
print("Classification Report (GridSearchCV):")
print(classification_report(y_test, y_pred_gs))