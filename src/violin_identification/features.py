import numpy as np
from tqdm.auto import tqdm

from core import Recording


def get_features_and_labels(feature_extractor, df):
    """
    Takes an instantiated feature extractor and a DataFrame of recordings.
    Returns the X feature matrix and y label array ready for sklearn.
    """
    X = []
    y = []

    feature_name = feature_extractor.__class__.__name__

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {feature_name}"):
        rec = Recording.from_dataframe(row)

        _, feature_vector = feature_extractor(rec)

        X.append(feature_vector)
        y.append(rec.violin)

    return np.array(X), np.array(y)
