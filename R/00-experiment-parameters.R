library(dplyr)
library(tibble)
library(tidyr)

ntrials = 18
start = 29.5
sigma = 2.08
interval.width = 0.5 # for study 1 only
optimal_temperature = qnorm(1 - 1/5, 32, 2.08)
size = 7
ngroups = ntrials - size + 1

set.seed(123)

xlims = c(20, 46)

# for study 1
sim_dist = tibble(
  index = 1:ntrials,
  mu = seq(start, start + interval.width*(ntrials-1), length.out = ntrials),
  sd = c(rnorm(8, sigma, 0.15), sigma, sigma, rnorm(ntrials - 10, sigma, 0.15))
) |> 
  mutate(dist = map2(mu, sd, dist_normal))

sim_dist.pnorm = sim_dist |> 
  mutate(
    x = map(index, ~ seq(xlims[1], xlims[2], by = 0.01)),
    y = map2(dist, x, ~ cdf(.x, .y)[[1]])
  ) |>
  unnest(c(x, y))

sim_dist.groups = tibble(
  group = 1:ngroups
) |> 
  mutate(
    data = list(sim_dist),
    data = map2(data, group, ~ filter(.x, index >= .y & index < .y + size))
  )

exp.design = rbind(
  sim_dist |> 
    group_by(index) |> 
    mutate(block = "single"),
  sim_dist.groups |> 
    unnest(c(data)) |> 
    select(-index) |> 
    rename(index = group) |> 
    mutate(block = "multiple")
) |> 
  mutate(p_true = map_dbl(dist, ~ cdf(.x, 32))) |> 
  group_by(block, index) |> 
  summarise(mu = list(mu), sd = list(sd), dist = list(dist), p_true = list(p_true), .groups = "keep") |> 
  mutate(
    .upper = map_dbl(p_true, min),
    .lower = map_dbl(p_true, max)
  )

# for study 2
set.seed(123)
size = 20

random_forecasts = truncnorm::rtruncnorm(size - 2, a = -0.7, b = 0.7, 0, 0.25) + 1.5

sim_dist.study2 = tibble(
  group = 1:ngroups
) |> 
  mutate(
    index = list(c(1:size)),
    mu = map(group, ~ c(0, random_forecasts, 3)),
    sd = map(group, ~ c(2.08, rnorm(size - 2, 2, 0.15), 2.08))
  ) |> 
  unnest(c(index, mu, sd))

