import argparse
import gc
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from envs import get_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="jbw")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    wall_time = np.random.randn(args.seeds, args.timesteps)
    memory = np.random.randn(args.seeds, args.timesteps)

    gc.collect()
    time.sleep(10)

    for seed in range(args.seeds):
        env = get_env(args.env)()
        env.reset(seed=seed)
        total_time = 0
        for step in tqdm(range(args.timesteps)):
            start = datetime.now()
            env.step(0)
            end = datetime.now()
            mem = psutil.virtual_memory()
            memory[seed, step] = mem.used
            total_time += (end - start).total_seconds()
            wall_time[seed, step] = total_time

        df = pd.DataFrame(
            {
                "wall_time": wall_time.flatten(),
                "memory": memory.flatten(),
                "seed": np.repeat(np.arange(args.seeds), args.timesteps),
                "step": np.tile(np.arange(args.timesteps), args.seeds),
            }
        )
        df.to_csv(results_dir / f"{args.env}.csv.gz", index=False)
        del df
        gc.collect()
        time.sleep(10)


if __name__ == "__main__":
    main()
