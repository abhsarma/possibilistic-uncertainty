// model 1
// estimates hierarchical effects for the linear-in-logit model
// considers only single forecasts
functions {
 /* compute correlated group-level effects
  * Args:
  *   z: matrix of unscaled group-level effects
  *   SD: vector of standard deviation parameters
  *   L: cholesky factor correlation matrix
  * Returns:
  *   matrix of scaled group-level effects
  */
  matrix scale_r_cor(matrix z, vector SD, matrix L) {
    // r is stored in another dimension order than z
    return transpose(diag_pre_multiply(SD, L) * z);
  }
}

// generated with brms 2.21.0
data {
  int<lower=1> N;            // total number of observations
  array[N] int Y;            // response variable
  vector[N] X;               // (actual probabilities) covariates for non-linear functions
  vector[N] Z;               // vector of 1s
  array[N] int<lower=1> pid;  // grouping indicator per observation
  int<lower=1> n_pid;         // number of grouping levels
  int<lower=1> M;             // number of coefficients per level
}
transformed data {
}
parameters {
  real b_alpha;               // regression coefficients
  real b_beta;                // regression coefficients
  vector<lower=0>[M] sd_1;    // group-level standard deviations
  array[M] vector[n_pid] z_1; // standardized group-level effects
  vector<lower=0>[M] sd_2;    // group-level standard deviations
  array[M] vector[n_pid] z_2; // standardized group-level effects
}
transformed parameters {
  vector[n_pid] r_alpha;      // actual group-level effects
  vector[n_pid] r_beta;       // actual group-level effects
  r_alpha = (sd_1[1] * (z_1[1]));
  r_beta  = (sd_2[1] * (z_2[1]));
}
model {
  //priors 
  target += normal_lpdf(b_alpha | 0, 1);
  target += normal_lpdf(b_beta | 1, 1);
  target += normal_lpdf(sd_1 | 0, 1) - 1 * normal_lccdf(0 | 0, 1);
  target += normal_lpdf(sd_2 | 0, 1) - 1 * normal_lccdf(0 | 0, 1);
  target += std_normal_lpdf(z_1[1]);
  target += std_normal_lpdf(z_2[1]);
  
  // likelihood including constants
  // initialize predictor terms
  vector[N] alpha = rep_vector(0.0, N);
  vector[N] beta = rep_vector(0.0, N);
  vector[N] mu;
  alpha += Z * b_alpha;
  beta += Z * b_beta;
  
  for (n in 1:N) {
    // add more terms to the linear predictor
    beta[n] += r_beta[pid[n]] * Z[n];
    // compute non-linear predictor values
    mu[n] = (alpha[n] + beta[n] * (logit(X[n]) - logit(0.2)));
  }
  target += bernoulli_logit_lpmf(Y | mu);
}
generated quantities {
}