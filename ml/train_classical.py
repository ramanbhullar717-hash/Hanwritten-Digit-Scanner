"""
train_classical.py - Training, Evaluation & Serialization for Classical ML Models
Trains:
1. Multi-Class Logistic Regression (Softmax link, Cross-Entropy)
2. K-Nearest Neighbors (k=5 with reference prototypes for Explainable AI)
3. K-Means Clustering (k=20 handwriting stroke archetypes)

Exports metrics, confusion reports, and serialized .joblib model bundles.
"""

import os
import sys
import time
import json
import numpy as np

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

try:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    SKLEARN_READY = True
except ImportError:
    SKLEARN_READY = False
    print("Warning: scikit-learn or joblib not installed in current environment.")


def load_mnist_data(max_train: int = 15000, max_test: int = 3000):
    """
    Loads MNIST dataset using torchvision or falls back to synthetic data.
    Limits samples to ensure training finishes in under 30 seconds.
    """
    print(f"Loading MNIST dataset (Train: {max_train}, Test: {max_test})...")
    data_dir = os.path.join(CURRENT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

        x_train = train_ds.data.numpy()[:max_train].reshape(max_train, 784).astype(np.float32) / 255.0
        y_train = train_ds.targets.numpy()[:max_train].astype(int)

        x_test = test_ds.data.numpy()[:max_test].reshape(max_test, 784).astype(np.float32) / 255.0
        y_test = test_ds.targets.numpy()[:max_test].astype(int)

        return x_train, y_train, x_test, y_test
    except Exception as e:
        print(f"Notice: Torchvision MNIST download skipped ({e}). Generating representative synthetic data...")
        np.random.seed(42)
        x_train = np.random.rand(max_train, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, size=max_train)
        x_test = np.random.rand(max_test, 784).astype(np.float32)
        y_test = np.random.randint(0, 10, size=max_test)
        return x_train, y_train, x_test, y_test


def train_and_export():
    if not SKLEARN_READY:
        print("Cannot train: Scikit-Learn and Joblib are required.")
        return

    saved_models_dir = os.path.join(BACKEND_DIR, "saved_models")
    outputs_dir = os.path.join(CURRENT_DIR, "outputs")
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    x_train, y_train, x_test, y_test = load_mnist_data()

    print("\n=======================================================")
    print(" 1. Training Multi-Class Logistic Regression (Softmax)")
    print("=======================================================")
    t0 = time.time()
    lr = LogisticRegression(max_iter=100, solver="lbfgs", multi_class="multinomial", random_state=42)
    lr.fit(x_train, y_train)
    lr_time = time.time() - t0
    y_pred_lr = lr.predict(x_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"  ✓ Logistic Regression trained in {lr_time:.2f}s | Test Accuracy: {acc_lr * 100:.2f}%")

    lr_path = os.path.join(saved_models_dir, "logistic_regression.joblib")
    joblib.dump(lr, lr_path)
    print(f"  ✓ Saved to: {lr_path}")

    print("\n=======================================================")
    print(" 2. Training K-Nearest Neighbors (k=5) + XAI Prototypes")
    print("=======================================================")
    t0 = time.time()
    # Use 5,000 reference prototypes for instant sub-10ms lookup
    ref_count = min(5000, len(x_train))
    x_ref = x_train[:ref_count]
    y_ref = y_train[:ref_count]

    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    knn.fit(x_ref, y_ref)
    knn_time = time.time() - t0
    y_pred_knn = knn.predict(x_test[:1000])  # evaluate on 1000 test items
    acc_knn = accuracy_score(y_test[:1000], y_pred_knn)
    print(f"  ✓ KNN trained in {knn_time:.2f}s | Test Accuracy: {acc_knn * 100:.2f}%")

    knn_bundle = {
        "model": knn,
        "reference_samples": x_ref,
        "reference_labels": y_ref
    }
    knn_path = os.path.join(saved_models_dir, "knn_model.joblib")
    joblib.dump(knn_bundle, knn_path)
    print(f"  ✓ Saved to: {knn_path}")

    print("\n=======================================================")
    print(" 3. Training K-Means Clustering (k=20 Archetypes)")
    print("=======================================================")
    t0 = time.time()
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    kmeans.fit(x_train[:8000])
    kmeans_time = time.time() - t0

    # Determine dominant digit label for each cluster
    cluster_assignments = kmeans.labels_
    dominant_labels = {}
    for c in range(20):
        members = y_train[:8000][cluster_assignments == c]
        if len(members) > 0:
            dominant_labels[c] = int(np.bincount(members).argmax())
        else:
            dominant_labels[c] = c % 10

    print(f"  ✓ K-Means fitted in {kmeans_time:.2f}s across 20 handwriting style clusters.")

    kmeans_bundle = {
        "kmeans": kmeans,
        "centers": kmeans.cluster_centers_,
        "dominant_labels": dominant_labels
    }
    kmeans_path = os.path.join(saved_models_dir, "kmeans_clusterer.joblib")
    joblib.dump(kmeans_bundle, kmeans_path)
    print(f"  ✓ Saved to: {kmeans_path}")

    # Generate Evaluation Report
    report = {
        "logistic_regression": {
            "accuracy": float(acc_lr),
            "train_time_sec": float(lr_time),
            "classification_report": classification_report(y_test, y_pred_lr, output_dict=True)
        },
        "knn": {
            "accuracy": float(acc_knn),
            "train_time_sec": float(knn_time)
        }
    }
    report_file = os.path.join(outputs_dir, "classical_metrics.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Performance metrics saved to {report_file}")
    print("\n=======================================================")
    print(" Classical ML Models Export Complete! ")
    print("=======================================================")


if __name__ == "__main__":
    train_and_export()
