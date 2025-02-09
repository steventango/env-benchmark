#!/bin/bash
python benchmark.py --env forager --timesteps 10000000 --subsample 10000 --seed 0
python benchmark.py --env forager --timesteps 10000000 --subsample 10000 --seed 1
python benchmark.py --env forager --timesteps 10000000 --subsample 10000 --seed 2
python benchmark.py --env forager --timesteps 10000000 --subsample 10000 --seed 3
python benchmark.py --env forager --timesteps 10000000 --subsample 10000 --seed 4
python benchmark.py --env jbw --timesteps 10000000 --subsample 10000 --seed 0
python benchmark.py --env jbw --timesteps 10000000 --subsample 10000 --seed 1
python benchmark.py --env jbw --timesteps 10000000 --subsample 10000 --seed 2
python benchmark.py --env jbw --timesteps 10000000 --subsample 10000 --seed 3
python benchmark.py --env jbw --timesteps 10000000 --subsample 10000 --seed 4
python plot.py
