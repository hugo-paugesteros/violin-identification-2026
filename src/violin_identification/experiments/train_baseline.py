import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sklearn.neural_network
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.metrics import make_scorer, cohen_kappa_score

from ..loader import load_recordings
from ..features import get_features_and_labels
from core import Recording, LTAS, LTCC, MFCC


def run_baseline_evaluation(recordings, output_path):
    """Runs the 10-iteration CV baseline evaluation for all datasets."""
    features = [LTAS(), LTCC(), MFCC()]
    classifiers = [
        KNeighborsClassifier(n_neighbors=21, weights="distance"),
        SVC(kernel="linear"),
    ]
    datasets = ["bilbao", "cnsm", "villefavard2026"]

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
        "kappa": make_scorer(cohen_kappa_score),
    }

    n_iters = 10
    cv = StratifiedShuffleSplit(n_splits=n_iters, test_size=0.2, random_state=42)
    results = []

    for dataset in datasets:
        data = recordings.query("dataset == @dataset")
        for feature in features:
            X, y = get_features_and_labels(feature, data)
            for clf in classifiers:
                scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                for i in range(n_iters):
                    iteration_results = {
                        "dataset": dataset,
                        "feature": feature.__class__.__name__,
                        "classifier": clf.__class__.__name__,
                        "iteration": i,
                    }
                    for metric_name in scoring.keys():
                        iteration_results[metric_name] = scores[f"test_{metric_name}"][
                            i
                        ]
                    results.append(iteration_results)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Baseline results successfully saved to {output_path}")


if __name__ == "__main__":
    from violin_identification.loader import load_recordings

    df = load_recordings()
    run_baseline_evaluation(df, output_path="data/interim/baseline_results.csv")
