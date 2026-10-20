import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import OUT, save


def run():
    datasets = {
        "linear": make_classification(
            n_samples=300,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=2,
            random_state=20,
        ),
        "moons": make_moons(n_samples=300, noise=0.2, random_state=20),
    }
    result = {}
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for row, (name, (X, y)) in enumerate(datasets.items()):
        a, b = train_test_split(
            np.arange(len(X)), test_size=0.25, stratify=y, random_state=20
        )
        result[name] = {}
        for col, kernel in enumerate(["linear", "rbf"]):
            model = make_pipeline(
                StandardScaler(),
                SVC(kernel=kernel, C=1 if kernel == "linear" else 10, gamma="scale"),
            ).fit(X[a], y[a])
            pred = model.predict(X[b])
            result[name][kernel] = dict(
                accuracy=float(accuracy_score(y[b], pred)),
                confusion=confusion_matrix(y[b], pred).tolist(),
                test_rows=len(b),
            )
            gx, gy = np.meshgrid(
                np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100),
            )
            axes[row, col].contourf(
                gx,
                gy,
                model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape),
                alpha=0.2,
                cmap="coolwarm",
            )
            axes[row, col].scatter(X[b, 0], X[b, 1], c=y[b], cmap="coolwarm", s=13)
            axes[row, col].set_title(name + " / " + kernel)
    fig.tight_layout()
    fig.savefig(OUT / "boundaries.png")
    plt.close(fig)
    return save(result)


if __name__ == "__main__":
    run()
