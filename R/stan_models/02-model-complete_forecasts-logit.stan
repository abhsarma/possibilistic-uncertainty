// model 3
// estimates hierarchical effects for the linear-in-logit model
// estimates hierarchical effects for the optimism index parameter
// considers both single and multiple forecasts as input data
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
  
  vector[N] P_upper;               // (upper bound of p-box) population-level design matrix
  vector[N] P_lower;               // (lower bound of p-box) population-level design matrix 
  
  array[N] int is_single;          // is the response from the CDF block or p-box block of trials
  int<lower=1> n_pid;              // number of participants
  array[N] int pid;                // participant indicator per observation
  int<lower=1> M;                  // number of coefficients per group (2)
}
transformed data {
}
parameters {
  real b_beta;                     // regression coefficients
  real b_alpha;
  real Intercept_gamma;
  
  vector<lower=0>[M] sd;           // group-level standard deviations
  matrix[M, n_pid] z;              // standardized group-level effects
  cholesky_factor_corr[M] L;       // cholesky factor of correlation matrix
  
  real<lower=0> sd_gamma;          // group-level standard deviations
  vector[n_pid] z_gamma;           // standardized group-level effects
}
transformed parameters {
  matrix[n_pid, M] r;
  r = scale_r_cor(z, sd, L);
  vector[n_pid] r_alpha;
  vector[n_pid] r_beta;
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
    target += normal_lpdf(Intercept_gamma | 0, 1);
    target += lkj_corr_cholesky_lpdf(L | 2);
    target += normal_lpdf(sd | 0, 1) - 2 * normal_lccdf(0 | 0, 1);
    target += normal_lpdf(to_vector(z) | 0, 1);
    target += normal_lpdf(sd_gamma | 0, 1) - 2 * normal_lccdf(0 | 0, 1);
    target += normal_lpdf(z_gamma | 0, 1);
  }
  
  // likelihood including constants
  // initialize linear predictor terms
  vector[N] gamma = rep_vector(0.0, N);
  vector[N] mu_single = rep_vector(0.0, N);       
  vector[N] mu_multiple = rep_vector(0.0, N);
  
  real c = logit(0.2);
  
  profile("linear parameters fixed effects") {
    gamma += Intercept_gamma;
    for (n in 1:N) {
      gamma[n] += r_gamma[pid[n]];
    }
    
    gamma = inv_logit(gamma);
    mu_single += b_alpha + (logit(P_upper) - c) * b_beta;
    mu_multiple += b_alpha + (logit((rep_vector(1, N) - gamma) .* P_lower + gamma .* P_upper) - c) * b_beta;
  }
  
  profile("linear parameters random effects") {
    // likelihood of the mixture model
    for (n in 1:N) {
      mu_single[n] += r_alpha[pid[n]] + r_beta[pid[n]] * (logit(P_upper[n]) - c);
      mu_multiple[n] += r_alpha[pid[n]] + r_beta[pid[n]] * (logit((1 - gamma[n]) * P_lower[n] + gamma[n] * P_upper[n]) - c);
    }
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