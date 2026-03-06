import argparse
from datetime import datetime
from math import ceil
from pathlib import Path

import jax.numpy as jnp
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scan-steps", type=int, default=1)
    parser.add_argument("--result-dir", type=str, default="results")
    args = parser.parse_args()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    length = ceil((args.timesteps) / args.subsample) + 1
    wall_time = np.random.randn(length)
    memory = np.random.randn(length)

    total_time = 0
    process = psutil.Process()
    mem_initial = process.memory_info().rss
    start = datetime.now()
    env = get_env(args.env)()
    env.reset(seed=args.seed)
    end = datetime.now()
    memory[0] = process.memory_info().rss
    total_time += (end - start).total_seconds()
    wall_time[0] = total_time
    use_scan = args.scan_steps > 1 and hasattr(env, "scan_step")
    step = 0
    with tqdm(total=args.timesteps) as pbar:
        while step < args.timesteps:
            prev_step = step
            if use_scan:
                n = min(args.scan_steps, args.timesteps - step, args.subsample)
                actions = jnp.zeros(n, dtype=jnp.int32)
                start = datetime.now()
                env.scan_step(actions)
                end = datetime.now()
                step += n
                pbar.update(n)
            else:
                start = datetime.now()
                env.step(0)
                end = datetime.now()
                step += 1
                pbar.update(1)
            total_time += (end - start).total_seconds()
            for s in range(prev_step + 1, step + 1):
                if s % args.subsample == 0:
                    index = s // args.subsample
                    memory[index] = process.memory_info().rss
                    wall_time[index] = total_time

    memory -= mem_initial
    df = pd.DataFrame(
        {
            "wall_time": wall_time.flatten(),
            "memory": memory.flatten(),
            "seed": np.full(length, args.seed),
            "step": np.arange(0, args.timesteps + 1, args.subsample),
        }
    )
    df.to_csv(results_dir / f"{args.env}.{args.seed}.csv.gz", index=False)


if __name__ == "__main__":
    main()
