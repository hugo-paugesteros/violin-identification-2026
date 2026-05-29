import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


def plot_listening_test(df: pd.DataFrame):
    df = df.copy()

    def parse_audio(filename):
        # Extracts Player, Violin, and Excerpt string (e.g., 'P1', 'VA', 'E1.3.0')
        match = re.search(r"(P\d)-(V[A-Z])-(E.*?)\.wav", str(filename))
        if match:
            return match.group(1), match.group(2), match.group(3)
        return None, None, None

    # 1. Parse conditions
    df["A_player"], df["A_violin"], df["A_excerpt"] = zip(*df["A"].apply(parse_audio))
    df["B_player"], df["B_violin"], df["B_excerpt"] = zip(*df["B"].apply(parse_audio))
    df["X_player"], df["X_violin"], df["X_excerpt"] = zip(*df["X"].apply(parse_audio))

    # 2. Establish Ground Truth (Which violin does X match?)
    def get_correct_answer(row):
        if row["X_violin"] == row["A_violin"]:
            return "A"
        if row["X_violin"] == row["B_violin"]:
            return "B"
        return None

    df["ground_truth"] = df.apply(get_correct_answer, axis=1)
    df["correct"] = (df["result"] == df["ground_truth"]).astype(int)

    # 3. Determine Excerpt Base (e.g., extracting 'E1' from 'E1.3.0')
    def get_base(e):
        if not isinstance(e, str):
            return e
        match = re.match(r"E\d", e)
        return match.group(0) if match else e

    df["same_player"] = (df["X_player"] == df["A_player"]) & (
        df["X_player"] == df["B_player"]
    )
    df["same_excerpt"] = (
        df["X_excerpt"].apply(get_base) == df["A_excerpt"].apply(get_base)
    ) & (df["X_excerpt"].apply(get_base) == df["B_excerpt"].apply(get_base))

    # 4. Generate grouping labels
    def categorize(row):
        p = "Same player" if row["same_player"] else "Different players"
        e = "same excerpt" if row["same_excerpt"] else "different excerpts"
        return f"{p},\n{e}"

    df["category"] = df.apply(categorize, axis=1)

    # Override base categories for exact identical audio vs different takes
    df.loc[(df["X"] == df["A"]) | (df["X"] == df["B"]), "category"] = (
        "Identical Audio\n(Exact same wav)"
    )
    print(df.sort_values(by=["listener", "id"]).head(15))
    # df.loc[
    #     ((df["X"] != df["A"]) & (df["X"] != df["B"])) & (df["same_excerpt"]), "category"
    # ] = "Different Takes\n(Same Excerpt)"

    # 5. Calculate Means and 95% Confidence Intervals
    summary = (
        df.groupby("category")["correct"].agg(["mean", "count", "std"]).reset_index()
    )
    summary["ci"] = 1.96 * summary["std"] / np.sqrt(summary["count"])
    summary["ci"] = summary["ci"].fillna(
        0
    )  # Handle NaN if a category only has 1 sample (std is undefined)

    # Calculate Overall Average
    overall_mean = df["correct"].mean()
    overall_std = df["correct"].std()

    avg_row = pd.DataFrame(
        [
            {
                "category": "Average Score",
                "mean": overall_mean,
                "count": len(df),
                "std": overall_std,
                "ci": 1.96 * overall_std / np.sqrt(len(df)) if len(df) > 0 else 0,
            }
        ]
    )

    summary = pd.concat([summary, avg_row], ignore_index=True)

    # 6. Plotting
    plt.figure(figsize=(12, 7))

    # Setup color palette (loops if there are more categories than colors)
    base_colors = [
        "#4C72B0",
        "#55A868",
        "#C44E52",
        "#8172B2",
        "#CCB974",
        "#64B5CD",
        "#8C8C8C",
    ]
    colors = [base_colors[i % len(base_colors)] for i in range(len(summary))]

    bars = plt.bar(
        summary["category"],
        summary["mean"],
        yerr=summary["ci"],
        capsize=8,
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )

    plt.axhline(0.5, color="red", linestyle="--", label="Chance Level (50%)")
    plt.ylim(0, 1.2)

    plt.ylabel("Accuracy Score (Mean + 95% CI)", fontsize=12)
    plt.title("Violin Identification Task Accuracy by Condition", fontsize=14)


def plot_listening_test(df: pd.DataFrame):
    df = df.query("id < 16").copy()

    category_map = {
        0: "Same Excerpt\nDifferent Player",
        1: "Same Excerpt\nDifferent Player",
        2: "Same Excerpt\nDifferent Player",
        3: "Same Excerpt\nSame Player\nSame Take",
        4: "Same Excerpt\nSame Player",
        5: "Same Excerpt\nSame Player",
        6: "Different Excerpt\nSame Player",
        7: "Different Excerpt\nSame Player",
        8: "Same Excerpt\nDifferent Player",
        9: "Same Excerpt\nDifferent Player",
        10: "Same Excerpt\nDifferent Player",
        11: "Same Excerpt\nSame Player\nSame Take",
        12: "Same Excerpt\nSame Player",
        13: "Same Excerpt\nSame Player",
        14: "Different Excerpt\nSame Player",
        15: "Different Excerpt\nSame Player",
    }

    df["category"] = df["id"].map(category_map)

    def parse_audio(filename):
        # Extracts Player, Violin, and Excerpt string (e.g., 'P1', 'VA', 'E1.3.0')
        match = re.search(r"(P\d)-(V[A-Z])-(E.*?)\.wav", str(filename))
        if match:
            return match.group(1), match.group(2), match.group(3)
        return None, None, None

    # 1. Parse conditions
    df["A_player"], df["A_violin"], df["A_excerpt"] = zip(*df["A"].apply(parse_audio))
    df["B_player"], df["B_violin"], df["B_excerpt"] = zip(*df["B"].apply(parse_audio))
    df["X_player"], df["X_violin"], df["X_excerpt"] = zip(*df["X"].apply(parse_audio))

    # 2. Establish Ground Truth (Which violin does X match?)
    def get_correct_answer(row):
        if row["X_violin"] == row["A_violin"]:
            return "A"
        if row["X_violin"] == row["B_violin"]:
            return "B"
        return None

    df["ground_truth"] = df.apply(get_correct_answer, axis=1)
    df["correct"] = (df["result"] == df["ground_truth"]).astype(int)

    listener_performance = (
        df.groupby(["listener", "category"])["correct"].mean().reset_index()
    )

    overall_per_listener = df.groupby("listener")["correct"].mean().reset_index()
    overall_per_listener["category"] = "Global\nAverage"

    # Merge the specific categories and global baseline together
    plot_data = pd.concat(
        [listener_performance, overall_per_listener], ignore_index=True
    )

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(
        data=plot_data,
        x="category",
        y="correct",
        # palette="viridis",
        errorbar="ci",  # Automatically computes 95% CI
        capsize=0.1,
    )

    # 3. Aesthetics
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance Level")
    plt.ylim(0, 1.1)
    plt.ylabel("Mean Accuracy")
    plt.xlabel("")
    plt.title(
        "Human Listening Test: Identification Accuracy", fontsize=14, weight="bold"
    )

    plt.show()


def plot_dataset(df, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(10, 4))

    my_cmap = plt.get_cmap("viridis")

    duration_sum = (
        df.groupby(["violin", "player"])["duration"]
        .sum()
        .unstack("player", fill_value=0)
    ) / 60.0

    ((ax1, ax2), (ax3, ax4)) = fig.subplots(
        2,
        2,
        sharey="row",
        sharex="col",
        gridspec_kw=dict(height_ratios=[1, 3], width_ratios=[3, 1]),
    )

    players = duration_sum.columns
    violins = duration_sum.index
    player_labels = np.arange(1, len(players) + 1)

    im = ax3.imshow(duration_sum.to_numpy(), aspect="auto", cmap="viridis")
    ax3.set_xticks(np.arange(len(players)), labels=player_labels)
    ax3.set_yticks(np.arange(len(violins)), labels=violins)
    ax3.set_xlabel("Player")
    ax3.set_ylabel("Violin")

    duration_per_player = duration_sum.sum(axis=0)
    norm = plt.Normalize(vmin=duration_per_player.min(), vmax=duration_per_player.max())
    ax1.bar(
        np.arange(len(players)),
        duration_per_player,
        align="center",
        color=my_cmap(norm(duration_per_player)),
    )
    ax1.set_ylabel("Rec. time (min)")
    ax1.tick_params(bottom=False)

    duration_per_violin = duration_sum.sum(axis=1)
    norm = plt.Normalize(vmin=duration_per_violin.min(), vmax=duration_per_violin.max())
    ax4.barh(
        np.arange(len(violins)),
        duration_per_violin,
        align="center",
        color=my_cmap(norm(duration_per_violin)),
    )
    ax4.set_xlabel("Rec. time (min)")
    ax4.tick_params(left=False)

    ax2.axis("off")
