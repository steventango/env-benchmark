#!/bin/bash
for i in {4..4}
do
    # python benchmark.py --env forager --timesteps 10000000 --subsample 10000 --seed $i
    python benchmark.py --env foragax --timesteps 10000000 --subsample 10000 --scan-steps 10000 --seed $i
    # python benchmark.py --env jbw --timesteps 10000000 --subsample 10000 --seed $i
    python plot.py
done
