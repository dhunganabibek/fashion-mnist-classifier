import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump
from training.ml_model import MulticlassLogisticRegression


def load_data(base: Path):
    train_csv = pd.read_csv(base / "fashion-mnist_train.csv", header=None, skiprows=1)
    test_csv = pd.read_csv(base / "fashion-mnist_test.csv", header=None, skiprows=1)
    X_train, y_train = train_csv.iloc[:, 1:], train_csv.iloc[:, 0].astype(int)
    X_test, y_test = test_csv.iloc[:, 1:], test_csv.iloc[:, 0].astype(int)
    return (
        np.asarray(X_train),
        np.asarray(y_train),
        np.asarray(X_test),
        np.asarray(y_test),
    )


def main():
    here = Path(__file__).parent
    X_train, y_train, X_test, y_test = load_data(here)

    model = MulticlassLogisticRegression(lr=0.01, epochs=1000, tol=1e-6)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(np.mean(y_pred == y_test))
    print(f"Accuracy: {acc:.4f}")

    out_path = here / "fashion-mnist-model.joblib"
    dump(model, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
