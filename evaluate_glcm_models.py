"""Reproducible grouped texture classification for the two supplied GLCM CSVs."""
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "Fiji": ROOT / "glcm_L16_d1_a0_1794925531103.csv",
    "Python": ROOT / "glcm_G16_d1_a0.csv",
}
FEATURES = ("contrast", "correlation", "asm", "homogeneity", "entropy")
LETTERS = "abcd"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    data = {}
    for row in rows:
        image = row["image"]
        match = re.search(r"-([abcd])-p\d+\.png$", image)
        if not match:
            raise ValueError(f"Unexpected image name: {image}")
        label = image.split("/")[0]
        if "label" in row and row["label"] != label:
            raise ValueError(f"Label mismatch: {image}")
        if "status" in row and row["status"] != "ok":
            raise ValueError(f"Invalid feature row: {image}")
        if image in data:
            raise ValueError(f"Duplicate image: {image}")
        data[image] = {"label": label, "group": match.group(1),
                       "x": [float(row[name]) for name in FEATURES]}
    return data


def standardize(x_train, x_eval):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1
    return (x_train - mean) / std, (x_eval - mean) / std


def predict_knn(train, labels, test, k):
    distances = cdist(test, train, metric="euclidean")
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    result = []
    for indices in order:
        votes = np.bincount(labels[indices], minlength=len(np.unique(labels)))
        tied = np.flatnonzero(votes == votes.max())
        result.append(min(tied, key=lambda cls: (distances[len(result), indices[labels[indices] == cls]].mean(), cls)))
    return np.array(result)


def fit_lda(train, labels, ridge, n_classes):
    means = np.stack([train[labels == cls].mean(axis=0) for cls in range(n_classes)])
    centered = train - means[labels]
    covariance = centered.T @ centered / (len(train) - n_classes)
    covariance += ridge * np.diag(np.diag(covariance))
    covariance += 1e-10 * np.eye(train.shape[1])
    weights = np.linalg.solve(covariance, means.T).T
    intercept = -0.5 * np.sum(means * weights, axis=1)
    return weights, intercept


def predict_lda(train, labels, test, ridge, n_classes):
    weights, intercept = fit_lda(train, labels, ridge, n_classes)
    return (test @ weights.T + intercept).argmax(axis=1)


def score(y, pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for actual, predicted in zip(y, pred):
        matrix[actual, predicted] += 1
    precision = np.divide(np.diag(matrix), matrix.sum(axis=0),
                          out=np.zeros(n_classes), where=matrix.sum(axis=0) != 0)
    recall = np.divide(np.diag(matrix), matrix.sum(axis=1),
                       out=np.zeros(n_classes), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(n_classes), where=precision + recall != 0)
    return {"accuracy": float(np.trace(matrix) / matrix.sum()),
            "precision_macro": float(precision.mean()),
            "recall_macro": float(recall.mean()),
            "f1_macro": float(f1.mean()),
            "per_class": [{"precision": float(p), "recall": float(r), "f1": float(f)}
                          for p, r, f in zip(precision, recall, f1)],
            "confusion_matrix": matrix.tolist()}


def run_model(x, y, group, model, candidates, n_classes):
    train_mask = group != "d"
    test_mask = ~train_mask
    cv = []
    for candidate in candidates:
        fold_scores = []
        for held_group in "abc":
            fit = train_mask & (group != held_group)
            valid = group == held_group
            x_fit, x_valid = standardize(x[fit], x[valid])
            pred = (predict_knn(x_fit, y[fit], x_valid, candidate) if model == "kNN"
                    else predict_lda(x_fit, y[fit], x_valid, candidate, n_classes))
            fold_scores.append(score(y[valid], pred, n_classes)["f1_macro"])
        cv.append({"parameter": candidate, "fold_f1_macro": fold_scores,
                   "mean_f1_macro": float(np.mean(fold_scores))})
    best = max(cv, key=lambda item: (item["mean_f1_macro"], -candidates.index(item["parameter"])))
    x_train, x_test = standardize(x[train_mask], x[test_mask])
    pred = (predict_knn(x_train, y[train_mask], x_test, best["parameter"]) if model == "kNN"
            else predict_lda(x_train, y[train_mask], x_test, best["parameter"], n_classes))
    return {"chosen_parameter": best["parameter"], "cv": cv,
            "test": score(y[test_mask], pred, n_classes)}


def main():
    rows = {source: read_rows(path) for source, path in SOURCES.items()}
    images = sorted(rows["Python"])
    if any(set(table) != set(images) for table in rows.values()):
        raise ValueError("CSV image sets do not match")
    labels = sorted({rows["Python"][image]["label"] for image in images})
    y = np.array([labels.index(rows["Python"][image]["label"]) for image in images])
    group = np.array([rows["Python"][image]["group"] for image in images])
    result = {"features": FEATURES, "excluded_feature": "energy (sqrt(asm))",
              "classes": labels, "n_images": len(images), "train_groups": "abc",
              "test_group": "d", "n_train": int(sum(group != "d")),
              "n_test": int(sum(group == "d")), "sources": {}}
    for source, table in rows.items():
        if any(table[image]["label"] != labels[y[i]] or table[image]["group"] != group[i]
               for i, image in enumerate(images)):
            raise ValueError(f"Label or group mismatch in {source}")
        x = np.array([table[image]["x"] for image in images])
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite feature in {source}")
        result["sources"][source] = {
            "kNN": run_model(x, y, group, "kNN", [1, 3, 5, 7, 11], len(labels)),
            "LDA": run_model(x, y, group, "LDA", [0.001, 0.01, 0.1, 1.0], len(labels)),
        }
        ridge = result["sources"][source]["LDA"]["chosen_parameter"]
        train = group != "d"
        ablation = {}
        for i, name in enumerate(FEATURES):
            reduced = np.delete(x, i, axis=1)
            x_train, x_test = standardize(reduced[train], reduced[~train])
            pred = predict_lda(x_train, y[train], x_test, ridge, len(labels))
            ablation[name] = score(y[~train], pred, len(labels))["f1_macro"]
        result["sources"][source]["LDA"]["drop_one_feature_test_f1_macro"] = ablation
    comparison = {}
    # Compare the six reported CSV features directly, including redundant Energy.
    with SOURCES["Fiji"].open(encoding="utf-8-sig", newline="") as handle:
        fiji_raw = {row["image"]: row for row in csv.DictReader(handle)}
    with SOURCES["Python"].open(encoding="utf-8-sig", newline="") as handle:
        python_raw = {row["image"]: row for row in csv.DictReader(handle)}
    for name in ("contrast", "correlation", "asm", "energy", "homogeneity", "entropy"):
        fiji = np.array([float(fiji_raw[image][name]) for image in images])
        python = np.array([float(python_raw[image][name]) for image in images])
        comparison[name] = {"fiji_mean": float(fiji.mean()),
                            "python_mean": float(python.mean()),
                            "mean_absolute_difference": float(np.mean(np.abs(fiji - python))),
                            "pearson_correlation": float(np.corrcoef(fiji, python)[0, 1])}
    result["feature_source_comparison"] = comparison
    path = ROOT / "glcm_classification_results.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for source, models in result["sources"].items():
        for model, info in models.items():
            print(source, model, info["chosen_parameter"],
                  {key: round(info["test"][key], 4) for key in
                   ("accuracy", "precision_macro", "recall_macro", "f1_macro")})
    print(path)


if __name__ == "__main__":
    main()
