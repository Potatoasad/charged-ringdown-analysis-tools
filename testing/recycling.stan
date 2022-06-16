data {
  int nobs;
  int nsamp;
  real chi_mean;
  real Q_mean;
  real chi_qs[nobs, nsamp, 2];
}

parameters {
  real mu_Q;
  real mu_chi;
  real<lower=0> sigma_Q;
  real<lower=0> sigma_chi;
}

model {
  mu_Q ~ normal(Q_mean, 1);
  sigma_Q ~ normal(0,1);
  mu_chi ~ normal(chi_mean, 1);
  sigma_chi ~ normal(0,1);

  /* Calculate expectation for each event */
  for (i in 1:nobs) {
    real logps[nsamp];
    real sigma_i = sqrt(sigma*sigma + bws[i]*bws[i]);
    for (j in 1:nsamp) {
      logps[j] = normal_lpdf(chis[i,j] | mu, sigma_i);
    }
    target += log_sum_exp(logps) - log(nsamp);
  }
}

generated quantities {
  /* We draw a sample from the Normal population of chi values at the value of
  /* mu and sigma we have fit. */
  real chi_pop;

  chi_pop = normal_rng(mu, sigma);
}

element
