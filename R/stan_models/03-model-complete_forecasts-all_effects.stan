// model 4
// estimates hierarchical effects for the linear-in-logit model
// estimates hierarchical effects for the optimism index parameter
// considers both single and multiple forecasts as input data
// estimates population-level effects for the gamma parameter
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

data {
  int<lower=1> N;                  // total number of observations
  array[N] int Y;                  // response variable
  
  int<lower=1> K;                  // number of coefficients for population level effects (vis: 2 levels)
  matrix[N, K] X;                  // [N x K] population-level design matrix
  
  vector[N] P_upper;               // (upper bound of p-box) population-level design matrix
  vector[N] P_lower;               // (lower bound of p-box) population-level design matrix 
  
  array[N] int is_single;          // is the response from the CDF block or p-box block of trials
  int<lower=1> n_pid;              // number of participants
  array[N] int pid;                // participant indicator per observation
  int<lower=1> M;                  // number of coefficients per group (3) (correlations in the intercept between alpha, beta and gamma)
}
transformed data {
}
parameters {
  real b_beta;                     // coefficients for degree of distortion for each level of vis
  real b_alpha;                    // coefficients for degree of bias for each level of vis
  vector[K] b_gamma;               // coefficients for the optimism parameter for each level of vis
  
  vector<lower=0>[M] sd;           // group-level standard deviations
  matrix[M, n_pid] z;              // standardized group-level effects
  cholesky_factor_corr[M] L;       // cholesky factor of correlation matrix
  
  real<lower=0> sd_gamma;          // group-level standard deviations
  vector[n_pid] z_gamma;           // standardized group-level effects
}
transformed parameters {
  matrix[n_pid, M] r;
  vector[n_pid] r_alpha;
  vector[n_pid] r_beta;
  
  r = scale_r_cor(z, sd, L);
  
  r_alpha = r[,1];
  r_beta = r[,2];
  
  vector[n_pid] r_gamma;
  r_gamma = (sd_gamma * z_gamma);
}
model {
  // priors
  profile("priors") {
    target += normal_lpdf(b_beta | 1, 1);
    target += normal_lpdf(b_alpha | 0, 1);
    target += normal_lpdf(b_gamma | 0, 1);
    target += lkj_corr_cholesky_lpdf(L | 2);
    target += normal_lpdf(sd | 0, 1) - 2 * normal_lccdf(0 | 0, 1);
    target += normal_lpdf(to_vector(z) | 0, 1);
    
    target += normal_lpdf(sd_gamma | 0, 2.5) - 2 * normal_lccdf(0 | 0, 2.5);
    target += normal_lpdf(z_gamma | 0, 1);
  }
  
  // likelihood including constants
  // initialize linear predictor terms
  vector[N] gamma = rep_vector(0.0, N);
  vector[N] beta = rep_vector(0.0, N);
  vector[N] alpha = rep_vector(0.0, N);
  
  vector[N] mu_single = rep_vector(0.0, N);       
  vector[N] mu_multiple = rep_vector(0.0, N);
  
  vector[N] Z = rep_vector(1.0, N);
  real c = logit(0.2);
  
  profile("linear parameters fixed and random effects") {
    alpha += Z * b_alpha;
    beta += Z * b_beta;
    gamma += X * b_gamma;
    
    for (n in 1:N) {
      alpha[n] += r_alpha[pid[n]] * Z[n];
      beta[n] += r_beta[pid[n]] * Z[n];
      gamma[n] += r_gamma[pid[n]] * Z[n];
    }
    
    gamma = inv_logit(gamma);
    mu_single += alpha + (logit(P_upper) - c) .* beta;
    mu_multiple += alpha + (logit((rep_vector(1, N) - gamma) .* P_lower + gamma .* P_upper) - c) .* beta;
  }
  
  profile("likelihood") {
    for (n in 1:N) {
      if (is_single[n]) {
        target += bernoulli_logit_lpmf(Y[n] | mu_single[n]);
      } else {
        target += bernoulli_logit_lpmf(Y[n] | mu_multiple[n]);
      }
    }
  }
}
generated quantities {
}