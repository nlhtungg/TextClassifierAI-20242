class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        # Khởi tạo trọng số
        self.w = [0.0 for _ in range(n_features)]
        # Gradient descent
        for _ in range(self.n_iters):
            dw = [0.0]*n_features
            db = 0.0
            for xi, yi in zip(X, y):
                y_pred = sum(wi*xi_j for wi, xi_j in zip(self.w, xi)) + self.b
                error = y_pred - yi
                for j in range(n_features):
                    dw[j] += (2/n_samples) * error * xi[j]
                db += (2/n_samples) * error
            # Cập nhật
            for j in range(n_features):
                self.w[j] -= self.lr * dw[j]
            self.b -= self.lr * db

    def predict(self, X):
        return [sum(wi*xi_j for wi, xi_j in zip(self.w, xi)) + self.b for xi in X]
