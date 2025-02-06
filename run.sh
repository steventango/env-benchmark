#!/bin/bash
python benchmark.py --env jbw --timesteps 10000000 --seeds 5
python benchmark.py --env forager --timesteps 10000000 --seeds 5
python plot.py
