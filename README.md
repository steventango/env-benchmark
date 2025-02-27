# env-benchmark

Scripts to benchmark the steps per second and memory usage of various
reinformcement learning environments.

## Usage

```bash
./run.sh
```

## Results

Below are the benchmark results for [Jellybean
World](https://github.com/eaplatanios/jelly-bean-world) and
[Forager](https://github.com/andnp/forager) environments, where the same action
is taken at each step, which results in the agent moving in a straight line.

### Steps per second
![results/benchmark_step.jpg](results/benchmark_steps.jpg)

Steps per second for each environment, higher is better. Error bars depict the
95% bootstrapped confidence interval of the mean.

Forager ran at 159879 steps per second, while Jellybean World ran at 487 steps
per second. Therefore, in this configuration, Forager is around 328 times faster
than Jellybean World. To complete 10 M timesteps, on average Jellybean World
took 5.7 hours, while Forager took only 1 minute. This amounts to wall time
savings of 5.7 hours per 10 M steps.

### Memory usage
![results/benchmark_memory.jpg](results/benchmark_memory.jpg)

Memory usage in GB as a function of the number of timesteps taken for each
environment, lower memory usage is better. Shaded regions represent the 95%
bootstrapped confidence interval of the mean. Individual runs are also plotted
with transparency.

Forager exhibited constant memory usage, while Jellybean World's memory usage
increased linearly with the number of timesteps. The memory usage for 10 M
timesteps, on average for Forager is only 0.1 GB of memory, which is only 0.8%
of Jelly Bean World's memory usage of 13.2 GB.
