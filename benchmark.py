import argparse
import gc
import time
from datetime import datetime
from math import ceil
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
    parser.add_argument("--subsample", type=int, default=10_000)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=str, default="results/epsilon-up")
    args = parser.parse_args()
    results_dir = Path(args.result_dir)
    results_dir.mkdir(exist_ok=True)

    length = ceil((args.timesteps) / args.subsample) + 1
    wall_time = np.random.randn(length)
    memory = np.random.randn(length)

    total_time = 0
    mem_initial = psutil.virtual_memory().used
    start = datetime.now()
    env = get_env(args.env)()
    env.reset(seed=args.seed)
    np.random.seed(args.seed)
    end = datetime.now()
    mem = psutil.virtual_memory()
    memory[0] = mem.used
    total_time += (end - start).total_seconds()
    wall_time[0] = total_time
    p = np.zeros(4)
    p[0] = 1 - args.epsilon
    p += args.epsilon / 4
    for step in tqdm(range(1, args.timesteps + 1)):
        action = np.random.choice(4, p=p)
        start = datetime.now()
        env.step(action)
        end = datetime.now()
        total_time += (end - start).total_seconds()
        if step % args.subsample == 0:
            index = step // args.subsample
            mem = psutil.virtual_memory()
            memory[index] = mem.used
            wall_time[index] = total_time

    memory -= mem_initial
    df = pd.DataFrame(
        {
            "wall_time": wall_time.flatten(),
            "memory": memory.flatten(),
            "seed": np.full(length, args.seed),
            "epsilon": np.full(length, args.epsilon),
            "step": np.arange(0, args.timesteps + 1, args.subsample),
        }
    )
    df.to_csv(results_dir / f"{args.env}_{args.epsilon}_{args.seed}.csv.gz", index=False)


if __name__ == "__main__":
    main()
