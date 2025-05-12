import random
from collections import Counter

# Cây quyết định nhị phân tối giản dùng Gini
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        # nếu đạt độ sâu tối đa hoặc quá ít mẫu, tạo lá
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'type': 'leaf', 'class': leaf_value}

        best_feat, best_thresh, best_gain = None, None, 0
        n_features = len(X[0])

        # tìm split tốt nhất
        for feat in range(n_features):
            values = set([xi[feat] for xi in X])
            for thresh in values:
                y_left, y_right = [], []
                X_left, X_right = [], []
                for xi, yi in zip(X, y):
                    if xi[feat] <= thresh:
                        X_left.append(xi); y_left.append(yi)
                    else:
                        X_right.append(xi); y_right.append(yi)
                if not X_left or not X_right:
                    continue
                gain = self._gini_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        if best_gain == 0:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'type': 'leaf', 'class': leaf_value}

        # chia nhánh
        X_l, y_l, X_r, y_r = [], [], [], []
        for xi, yi in zip(X, y):
            if xi[best_feat] <= best_thresh:
                X_l.append(xi); y_l.append(yi)
            else:
                X_r.append(xi); y_r.append(yi)

        left_sub = self.fit(X_l, y_l, depth+1)
        right_sub = self.fit(X_r, y_r, depth+1)
        return {
            'type': 'node',
            'feat': best_feat,
            'thresh': best_thresh,
            'left': left_sub,
            'right': right_sub
        }

    def _gini(self, y):
        m = len(y)
        counts = Counter(y)
        return 1 - sum((cnt/m)**2 for cnt in counts.values())

    def _gini_gain(self, parent, left, right):
        p = len(left)/len(parent)
        return self._gini(parent) - p*self._gini(left) - (1-p)*self._gini(right)

    def predict_one(self, xi, node):
        if node['type'] == 'leaf':
            return node['class']
        if xi[node['feat']] <= node['thresh']:
            return self.predict_one(xi, node['left'])
        else:
            return self.predict_one(xi, node['right'])

    def predict(self, X):
        return [self.predict_one(xi, self.tree) for xi in X]


class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, sample_size=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_size = sample_size
        self.trees = []

    def _sample(self, X, y):
        n = len(X) if self.sample_size is None else self.sample_size
        idxs = [random.randrange(len(X)) for _ in range(n)]
        return [X[i] for i in idxs], [y[i] for i in idxs]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_samp, y_samp = self._sample(X, y)
            tree = DecisionTree(self.max_depth, self.min_samples_split)
            tree.tree = tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        # vote đa số
        tree_preds = [tree.predict(X) for tree in self.trees]
        preds = []
        for i in range(len(X)):
            votes = [tree_preds[t][i] for t in range(self.n_trees)]
            preds.append(Counter(votes).most_common(1)[0][0])
        return preds
