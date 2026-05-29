import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sklearn.neural_network
from sklearn.svm import SVC
import sklearn.tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

from ..loader import load_recordings, filter_robust_cohort
from ..features import get_features_and_labels
from core import Recording, LTAS, LTCC, MFCC


def run_one_factor_evaluation(recordings, factor, output_path):
    datasets = ["cnsm", "villefavard2026"]
    feature = MFCC()
    classifier = SVC(kernel="linear")
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    base_metrics = {
        "accuracy": (accuracy_score, {}),
        "precision": (precision_score, {"average": "macro", "zero_division": 0}),
        "recall": (recall_score, {"average": "macro", "zero_division": 0}),
        "f1": (f1_score, {"average": "macro", "zero_division": 0}),
        "kappa": (cohen_kappa_score, {}),
    }

    # 2. Auto-generate the sklearn-compatible scorers for cross_validate (Test A)
    scoring = {
        name: make_scorer(func, **kwargs)
        for name, (func, kwargs) in base_metrics.items()
    }

    results = []
    for dataset in datasets:
        subset = recordings.query("dataset == @dataset")
        groups = subset[factor].unique()

        X_all, y_all = get_features_and_labels(feature, subset)
        groups_all = subset[factor].values

        for group in groups:
            print(group)
            mask_same = groups_all == group
            mask_diff = groups_all != group

            X_same, y_same = X_all[mask_same], y_all[mask_same]
            X_diff, y_diff = X_all[mask_diff], y_all[mask_diff]

            for i in range(10):
                X_same_train, X_same_test, y_same_train, y_same_test = train_test_split(
                    X_same, y_same, test_size=0.2, random_state=42 + i, stratify=y_same
                )

                # Train
                model = clone(classifier)
                model.fit(X_same_train, y_same_train)

                # Test A : same
                y_pred_same = model.predict(X_same_test)

                res_same = {
                    "dataset": dataset,
                    f"{factor}_trained_on": group,
                    "test_type": f"Same {factor.capitalize()}",
                    "iteration": i,
                }
                for metric_name, (func, kwargs) in base_metrics.items():
                    res_same[metric_name] = func(y_same_test, y_pred_same, **kwargs)
                results.append(res_same)

                # Test B : different
                y_pred_diff = model.predict(X_diff)

                res_diff = {
                    "dataset": dataset,
                    f"{factor}_trained_on": group,
                    "test_type": f"Different {factor.capitalize()}",
                    "iteration": i,
                }
                for metric_name, (func, kwargs) in base_metrics.items():
                    res_diff[metric_name] = func(y_diff, y_pred_diff, **kwargs)
                results.append(res_diff)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Baseline results successfully saved to {output_path}")


if __name__ == "__main__":
    from violin_identification.loader import load_recordings

    recordings = load_recordings()
    recordings = filter_robust_cohort(recordings)
    # run_one_factor_evaluation(
    #     recordings, "excerpt", output_path="data/interim/excerpts_results.csv"
    # )
    recordings = recordings[~recordings.player.isin(["David", "Mélanie", "Tanguy"])]
    run_one_factor_evaluation(
        recordings, "player", output_path="data/interim/players_results.csv"
    )
