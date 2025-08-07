import numpy as np

class ClassifierHead:
    def __init__(self, dim, num_classes=3):
        self.W_vocab = np.random.normal(0, np.sqrt(2 / dim), size=(dim, num_classes))
        self.b_vocab = np.zeros(num_classes)
        self.cached = {}

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def probabilities(self, X_cls):
        logits = X_cls @ self.W_vocab + self.b_vocab
        return self.softmax(logits), logits
  