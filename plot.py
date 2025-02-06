from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    env_mapping = {
        "jbw": "Jellybean World",
        "forager": "Forager",
    }
    csvs = Path("results").rglob("*.csv.gz")
    dfs = []
    for csv in csvs:
        df = pd.read_csv(csv)
        df["env"] = csv.stem.removesuffix(".csv")
        df["env"] = df["env"].map(env_mapping)
        df["memory"] = df.groupby("seed")["memory"].transform(lambda x: x - x.min())
        df["memory"] /= 1e9
        df = df[df["step"] % 1000 == 0]
        dfs.append(df)
    df = pd.concat(dfs)

    plt.figure()
    df_agg = df.groupby(["env", "seed"]).agg({"wall_time": "max"}).sort_values("wall_time")
    df_agg["fps"] = df["step"].max() / df_agg["wall_time"]
    sns.barplot(data=df_agg, x="env", y="fps", hue="env", capsize=0.2)
    sns.despine()
    plt.ylabel("Frames per second")
    plt.xlabel("Environment")
    plt.tight_layout()
    plt.savefig("results/benchmark_fps.pdf")

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
    plt.xlabel("Timesteps")
    plt.ylabel("Memory (GB)")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.legend(title=None)
    plt.tight_layout()
    plt.savefig("results/benchmark_memory.pdf")


if __name__ == "__main__":
    main()
