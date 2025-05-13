class GaussianNB:
    def __init__(self):
        self.classes = []
        self.priors = {}
        self.mean = {}
        self.var = {}

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.classes = set(y)
        data = defaultdict(list)
        for xi, yi in zip(X, y):
            data[yi].append(xi)
        for c, samples in data.items():
            self.priors[c] = len(samples) / n_samples
            features = list(zip(*samples))
            self.mean[c] = [sum(f)/len(f) for f in features]
            self.var[c] = [sum((val - m)**2 for val in f)/len(f) for f, m in zip(features, self.mean[c])]

    def _pdf(self, x, mean, var):
        eps = 1e-9
        coeff = 1.0 / math.sqrt(2.0 * math.pi * (var + eps))
        exponent = math.exp(-((x-mean)**2) / (2 * (var + eps)))
        return coeff * exponent

    def predict(self, X):
        preds = []
        for xi in X:
            posteriors = {}
            for c in self.classes:
                log_prob = math.log(self.priors[c])
                for i, x_val in enumerate(xi):
                    log_prob += math.log(self._pdf(x_val, self.mean[c][i], self.var[c][i]))
                posteriors[c] = log_prob
            preds.append(max(posteriors, key=posteriors.get))
        return preds