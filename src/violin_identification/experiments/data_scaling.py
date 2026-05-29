import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)

from ..loader import load_recordings, filter_robust_cohort
from ..features import get_features_and_labels
from core import MFCC


def run_data_scaling_evaluation(recordings, output_path):
    """
    Evaluates model performance across varying amounts of training data
    (number of players and excerpts per player).
    """
    datasets = ["bilbao", "cnsm", "villefavard2026"]
    feature = MFCC()
    classifier = SVC(kernel="linear")

    player_grid = [1, 2, 3, 4, 5, 6, 7]
    excerpt_grid = [1, 2, 3, 5, 10, 15]
    n_iterations = 15

    base_metrics = {
        "accuracy": (accuracy_score, {}),
        "precision": (precision_score, {"average": "macro", "zero_division": 0}),
        "recall": (recall_score, {"average": "macro", "zero_division": 0}),
        "f1": (f1_score, {"average": "macro", "zero_division": 0}),
        "kappa": (cohen_kappa_score, {}),
    }

    results = []

    for dataset in datasets:
        df_sub = recordings.query("dataset == @dataset")
        if df_sub.empty:
            continue

        X_all, y_all = get_features_and_labels(feature, df_sub)
        players_all = df_sub["player"].values

        # 1. Global Test Split (Stable Holdout Pool)
        X_pool, X_test, y_pool, y_test, p_pool, p_test = train_test_split(
            X_all, y_all, players_all, test_size=0.2, stratify=y_all, random_state=42
        )
        unique_players = np.unique(p_pool)

        # 2. Iterate over the Grid
        for n_players in player_grid:
            if n_players > len(unique_players):
                continue

            for n_excerpts in excerpt_grid:
                # 3. Create N Monte Carlo Splits
                for iteration in range(n_iterations):
                    # A. Sample Players
                    chosen_players = np.random.choice(
                        unique_players, n_players, replace=False
                    )

                    # B. Sample Excerpts per Player
                    train_idx = []
                    for p in chosen_players:
                        p_idx = np.where(p_pool == p)[0]
                        n_to_take = min(n_excerpts, len(p_idx))
                        if n_to_take > 0:
                            train_idx.extend(
                                np.random.choice(p_idx, n_to_take, replace=False)
                            )

                    if not train_idx:
                        continue

                    X_train, y_train = X_pool[train_idx], y_pool[train_idx]

                    # Safety Check: Must have at least 2 distinct violins to train
                    unique_train_violins = np.unique(y_train)
                    if len(unique_train_violins) < 2:
                        continue

                    # C. Train the Model
                    model = clone(classifier)
                    model.fit(X_train, y_train)

                    # # D. Test on Known Violins Only
                    # mask_known = np.isin(y_test, unique_train_violins)
                    # if sum(mask_known) == 0:
                    #     continue

                    # y_test_known = y_test[mask_known]
                    # y_pred_known = model.predict(X_test[mask_known])

                    y_pred = model.predict(X_test)

                    # E. Log Metrics
                    res = {
                        "dataset": dataset,
                        "n_players": n_players,
                        "n_excerpts_per_player": n_excerpts,
                        "iteration": iteration,
                    }
                    for name, (func, kwargs) in base_metrics.items():
                        res[name] = func(y_test, y_pred, **kwargs)

                    results.append(res)

    # 4. Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Data scaling results successfully saved to {output_path}")


if __name__ == "__main__":
    from violin_identification.loader import load_recordings, filter_robust_cohort

    recordings = load_recordings()
    recordings = filter_robust_cohort(recordings)

    run_data_scaling_evaluation(
        recordings, output_path="data/interim/scaling_results.csv"
    )
