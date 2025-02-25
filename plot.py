from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

gdm_colors = ["#4285f4", "#ff902a", "#34a853", "#ea4335", "#fbbc04", "#2daeb8"]


def main():
    env_mapping = {
        "jbw": "Jellybean World",
        "forager": "Forager",
    }
    csvs = Path("results").rglob("*.csv.gz")
    df = load_df(csvs, env_mapping)

    df_agg = df.groupby(["env", "seed"]).agg({"step": "max", "wall_time": "max", "memory": "max"}).reset_index()
    df_agg["sps"] = df_agg["step"] / df_agg["wall_time"]
    df_agg = df_agg.groupby("env").agg({"sps": "mean", "memory": "mean"}).reset_index()
    print(df_agg)

    plot_sps(df_agg)
    plot_memory(df)
    plot_wall_time(df)


def load_df(csvs: Iterable[Path], env_mapping: dict):
    dfs = []
    for csv in csvs:
        df = pd.read_csv(csv)
        df["env"] = csv.stem.split(".")[0]
        df["env"] = df["env"].map(env_mapping)
        df["memory"] /= 1e9
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.sort_values("env")
    return df


def plot_wall_time(df: pd.DataFrame):
    plt.figure(figsize=(3, 2))

    sns.lineplot(data=df, x="step", y="wall_time", hue="env", palette=gdm_colors, legend=False)
    sns.lineplot(
        data=df,
        x="step",
        y="wall_time",
        hue="env",
        palette=gdm_colors,
        alpha=0.4,
        units="seed",
        estimator=None,
        legend=False,
        errorbar=None,
    )
    sns.despine()
    plt.xlabel("Timestep")
    plt.ylabel("Wall time (s)")
    plt.gca().ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)

    for line, label in zip(plt.gca().get_lines(), df["env"].unique()):
        y = line.get_ydata()[-1]
        w = df["step"].max() - df["step"].min()
        h = df["wall_time"].max() - df["wall_time"].min()
        plt.text(
            df["step"].max() - 0.1 * w, y + 0.1 * h, label, color=line.get_color(), fontsize=10, va="center", ha="right"
        )

    plt.tight_layout()
    plt.savefig("results/benchmark_wall_time.pdf")
    plt.savefig("results/benchmark_wall_time.jpg")


def plot_memory(df: pd.DataFrame):
    plt.figure(figsize=(3, 2))

    sns.lineplot(data=df, x="step", y="memory", hue="env", palette=gdm_colors, legend=False)
    sns.lineplot(
        data=df,
        x="step",
        y="memory",
        hue="env",
        palette=gdm_colors,
        alpha=0.4,
        units="seed",
        estimator=None,
        legend=False,
        errorbar=None,
    )
    sns.despine()
    plt.xlabel("Timestep")
    plt.ylabel("Memory (GB)")
    plt.gca().ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)

    for line, label in zip(plt.gca().get_lines(), df["env"].unique()):
        y = line.get_ydata()[-1]
        w = df["step"].max() - df["step"].min()
        h = df["memory"].max() - df["memory"].min()
        plt.text(
            df["step"].max() - 0.1 * w, y + 0.1 * h, label, color=line.get_color(), fontsize=10, va="center", ha="right"
        )

    plt.tight_layout()
    plt.savefig("results/benchmark_memory.pdf")
    plt.savefig("results/benchmark_memory.jpg")


def plot_sps(df_agg: pd.DataFrame):
    plt.figure(figsize=(3, 2))
    sns.barplot(data=df_agg, x="env", y="sps", hue="env", palette=gdm_colors, capsize=0.2)
    sns.despine()
    plt.xlabel("Environment")
    plt.ylabel("Steps per second")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("results/benchmark_steps.pdf")
    plt.savefig("results/benchmark_steps.jpg")


if __name__ == "__main__":
    main()
