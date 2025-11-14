import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Stable softmax over axis=1, float64 for better numerical behavior.
    z: (n_samples, n_classes)
    returns: (n_samples, n_classes) with rows summing to 1
    """
    z = np.asarray(z, dtype=np.float64, copy=False)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    s = e.sum(axis=1, keepdims=True)
    s[s == 0.0] = 1.0
    return e / s


class MulticlassLogisticRegression:
    """
    Multinomial (softmax) logistic regression trained with batch gradient descent.
    Standardizes inputs using mean/std learned on training data.
    """

    def __init__(
        self, lr: float = 0.05, epochs: int = 1500, tol: float = 1e-6, l2: float = 1e-4
    ):
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.l2 = l2  # L2 regularization coefficient
        self.W_: np.ndarray | None = None  # (n_features, n_classes)
        self.classes_: np.ndarray | None = None
        self.mean: np.ndarray | None = None  # (n_features,)
        self.std: np.ndarray | None = None  # (n_features,)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MulticlassLogisticRegression":
        # Use float64 for training stability
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # Classes and inverse indices (y_idx maps each sample to its class index)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = int(self.classes_.shape[0])

        # Standardize with clamped std to avoid exploding features
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        std_floor = 1e-2  # more sensible than 1e-8 for images with many constant pixels
        self.std = np.where(std < std_floor, std_floor, std)
        Xn = (X - self.mean) / self.std

        # Initialize weights
        self.W_ = np.zeros((n_features, n_classes), dtype=np.float64)

        # One-hot labels
        Y = np.eye(n_classes, dtype=np.float64)[y_idx]  # (n_samples, n_classes)

        # Batch gradient descent with L2
        for _ in range(self.epochs):
            logits = Xn @ self.W_  # (n_samples, n_classes)
            probs = softmax(logits)  # (n_samples, n_classes)
            # Cross-entropy gradient + L2 on weights
            grad = (Xn.T @ (probs - Y)) / n_samples + self.l2 * self.W_

            # Update
            self.W_ -= self.lr * grad

            # Early stopping
            if np.linalg.norm(grad) < self.tol:
                break

        return self

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Model is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean) / self.std

    def predict_proba(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        if self.W_ is None:
            raise RuntimeError("Model is not fitted.")
        Xn = self._standardize(X)
        logits = Xn @ self.W_
        if temperature and temperature > 0 and temperature != 1.0:
            logits = logits / float(temperature)
        return softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
