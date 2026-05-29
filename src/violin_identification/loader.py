from pathlib import Path
import json
import pandas as pd

MAP = {
    "A": "V1",
    "B": "V2",
    "C": "V3",
    "D": "V4",
    "E": "V5",
    "F": "V6",
    "G": "V7",
    "H": "V8",
    "I": "V9",
    "J": "V13",
}


def load_villefavard_recordings(data_folder="."):
    """
    Scans the data_folder recursively for .txt and .json label files,
    parses them, pairs them with their corresponding .wav files, and
    returns a concatenated DataFrame.
    """
    all_labels = []
    base_path = Path(data_folder)

    for label_file in base_path.glob("*/*"):
        ext = label_file.suffix.lower()

        if ext not in (".txt", ".json"):
            continue

        if ext == ".txt":
            df = parse_audacity_labels(str(label_file))
        elif ext == ".json":
            df = parse_json_labels(str(label_file))

        wav_file = label_file.with_suffix(".wav")
        df["file"] = str(wav_file)

        if "violin" in df.columns:
            df["violin"] = df["violin"].replace(MAP)

        all_labels.append(df)

    if not all_labels:
        print(f"No .txt or .json files found in {data_folder}")
        return pd.DataFrame()

    return pd.concat(all_labels).reset_index(drop=True)


def parse_audacity_labels(filepath):
    labels = pd.read_csv(filepath, sep="\t")
    labels.columns = ["start", "end", "comments"]

    labels["duration"] = labels.end - labels.start
    labels[["violin", "player", "excerpt"]] = labels.comments.str.split(
        pat=";", expand=True
    )
    labels.drop("comments", axis=1, inplace=True)
    return labels


def parse_json_labels(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Failsafe for empty JSON files so downstream processing doesn't crash
    if not raw_data:
        return pd.DataFrame(
            columns=["start", "end", "duration", "violin", "player", "excerpt"]
        )

    parsed_data = []
    for item in raw_data:
        start = item.get("start", 0.0)
        end = item.get("end", 0.0)
        meta = item.get("data", {})

        parsed_data.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "violin": meta.get("violin", ""),
                "player": meta.get("player", ""),
                "excerpt": meta.get("excerpt", ""),
            }
        )

    return pd.DataFrame(parsed_data)


def load_listening_tests(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    files = path.rglob("*.json")

    data = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        for item in json_data["results"]:
            data.append(
                {
                    "listener": file.stem,
                    "id": item["test"]["id"],
                    "A": item["test"]["a"],
                    "B": item["test"]["b"],
                    "X": item["test"]["x"],
                    "result": item["result"],
                }
            )
    return pd.DataFrame(data)


def load_recordings() -> pd.DataFrame:
    # Bilbao
    bilbao = pd.read_pickle(
        "/home/hugo/Thèse/identification/data/processed/dataset_bilbao.pkl"
    )
    bilbao.rename({"type": "excerpt"}, axis=1, inplace=True)
    bilbao["dataset"] = "bilbao"
    bilbao = bilbao[bilbao.excerpt.isin(["scale", "free"])]

    # CNSM
    # TODO : reparse all files
    cnsm = pd.read_pickle(
        "/home/hugo/Thèse/identification/data/processed/dataset_cnsm.pkl"
    )
    cnsm.rename({"extract": "excerpt"}, axis=1, inplace=True)
    cnsm["dataset"] = "cnsm"
    cnsm = cnsm[~cnsm.violin.isin(["?", "rire"])]
    cnsm = cnsm[~cnsm.excerpt.isin(["?"])]
    cnsm.loc[(cnsm.excerpt == "sibelius") & (cnsm.session == 3), "excerpt"] = (
        "sibelius2"
    )

    # Villefavard 2026
    villefavard = load_villefavard_recordings("/home/hugo/Thèse/Data/Villefavard 2026/")
    villefavard["dataset"] = "villefavard2026"
    villefavard = villefavard[
        ~villefavard.violin.isin(["SMD", "V10", "V11", "V12", "V3", "V5"])
    ]

    villefavard = villefavard[
        villefavard.excerpt.isin(
            [
                "Scale",
                "Bach",
                "Mozart",
                "Tchai",
                "Sibelius",
                "Glazounov",
                "Glazounov2",
                "Free",
            ]
        )
    ]

    dataset = pd.concat([bilbao, cnsm, villefavard]).reset_index(drop=True)
    return dataset


def filter_robust_cohort(df, min_players=5):
    """
    Filters the dataset to only include highly represented violins and excerpts.
    """
    df_filtered = df.copy()

    # 1. Count unique players for each violin (grouped by dataset to be safe)
    # 'nunique' counts the number of distinct people, not just the number of takes
    players_per_violin = df_filtered.groupby(["dataset", "violin"])["player"].transform(
        "nunique"
    )

    # 2. Count unique players for each excerpt
    players_per_excerpt = df_filtered.groupby(["dataset", "excerpt"])[
        "player"
    ].transform("nunique")

    # 3. Create the boolean mask (> 5 means 6 or more. Use >= if you meant 5 or more)
    robust_mask = (players_per_violin > min_players) & (
        players_per_excerpt > min_players
    )

    # 4. Apply the mask
    df_clean = df_filtered[robust_mask]

    # Optional: Print a quick report to see how much data was dropped
    dropped = len(df) - len(df_clean)
    print(f"Original recordings: {len(df)}")
    print(f"Cleaned recordings:  {len(df_clean)} (Dropped {dropped})")

    return df_clean
