from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def load_df(csvs: Iterable[Path], env_mapping: dict):
    dfs = []
    for csv in csvs:
        df = pd.read_csv(csv)
        df["env"] = csv.stem.split(".")[0]
        df["env"] = df["env"].map(env_mapping)
        df["memory"] /= 1e9
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def plot_memory(df: pd.DataFrame):
    plt.figure()

    sns.lineplot(data=df, x="step", y="memory", hue="env")
    sns.lineplot(
        data=df,
        x="step",
        y="memory",
        hue="env",
        alpha=0.4,
        units="seed",
        estimator=None,
        legend=False,
        errorbar=None,
    )
    sns.despine()
    plt.xlabel("Timestep")
    plt.ylabel("Memory (GB)")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.legend(title=None)
    plt.tight_layout()
    plt.savefig("results/benchmark_memory.pdf")
    plt.savefig("results/benchmark_memory.jpg")


def plot_sps(df_agg: pd.DataFrame):
    plt.figure()
    sns.barplot(data=df_agg, x="env", y="sps", hue="env", capsize=0.2)
    sns.despine()
    plt.xlabel("Environment")
    plt.ylabel("Steps per second")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("results/benchmark_steps.pdf")
    plt.savefig("results/benchmark_steps.jpg")


if __name__ == "__main__":
    main()
