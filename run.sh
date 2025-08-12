#!/bin/bash
# python benchmark.py --env forager --timesteps 10000000 --subsample 10000 --seed 0
# python benchmark.py --env forager --timesteps 10000000 --subsample 10000 --seed 1
# python benchmark.py --env forager --timesteps 10000000 --subsample 10000 --seed 2
# python benchmark.py --env forager --timesteps 10000000 --subsample 10000 --seed 3
# python benchmark.py --env forager --timesteps 10000000 --subsample 10000 --seed 4
# python benchmark.py --env jbw --timesteps 10000000 --subsample 10000 --seed 0
# python benchmark.py --env jbw --timesteps 10000000 --subsample 10000 --seed 1
# python benchmark.py --env jbw --timesteps 10000000 --subsample 10000 --seed 2
# python benchmark.py --env jbw --timesteps 10000000 --subsample 10000 --seed 3
# python benchmark.py --env jbw --timesteps 10000000 --subsample 10000 --seed 4

# python plot.py --result-dir results

python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 1 --result-dir results/epsilon-up --seed 0
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.5 --result-dir results/epsilon-up --seed 0
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.25 --result-dir results/epsilon-up --seed 0
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.0 --result-dir results/epsilon-up --seed 0
python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 1 --result-dir results/epsilon-up --seed 1
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.5 --result-dir results/epsilon-up --seed 1
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.25 --result-dir results/epsilon-up --seed 1
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.0 --result-dir results/epsilon-up --seed 1
python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 1 --result-dir results/epsilon-up --seed 2
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.5 --result-dir results/epsilon-up --seed 2
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.25 --result-dir results/epsilon-up --seed 2
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.0 --result-dir results/epsilon-up --seed 2
python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 1 --result-dir results/epsilon-up --seed 3
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.5 --result-dir results/epsilon-up --seed 3
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.25 --result-dir results/epsilon-up --seed 3
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.0 --result-dir results/epsilon-up --seed 3
python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 1 --result-dir results/epsilon-up --seed 4
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.5 --result-dir results/epsilon-up --seed 4
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.25 --result-dir results/epsilon-up --seed 4
# python benchmark.py --env jbw --timesteps 100000 --subsample 10000 --epsilon 0.0 --result-dir results/epsilon-up --seed 4

python plot.py --result-dir results/epsilon-up
