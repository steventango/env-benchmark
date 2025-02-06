#!/bin/bash
python benchmark.py --env jbw --timesteps 1000001 --subsample 10000 --seeds 5
python benchmark.py --env forager --timesteps 1000001 --subsample 10000 --seeds 5
python plot.py
