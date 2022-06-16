functions {
  real pdf_truncated_normal(real x, real mu, real sigma, real a, real b) {
    real numerator;
    real denomenator;
    numerator = exp(normal_lpdf(x | mu, sigma));
    denomenator = normal_cdf(b, mu, sigma) - normal_cdf(a, mu, sigma);
    return numerator/denomenator;
  }

  real prior_conditioned_on_hyperprior(real r, real theta, real mu_r, real sigma_r, real mu_theta, real sigma_theta) {
    return pdf_truncated_normal(r, mu_r, sigma_r, 0, 1)*pdf_truncated_normal(theta, mu_theta, sigma_theta, 0, pi()/2)
  }
}

data {
  int nobs;
  int nsamp;

  real r_mean;
  real theta_mean;
  real r_thetas[nobs, nsamp, 2];
}

parameters {
  real mu_r;
  real mu_theta;
  real<lower=0> sigma_r;
  real<lower=0> sigma_theta;
}

model {
  mu_r ~ normal(r_mean, 1);
  sigma_r ~ normal(0,1);
  mu_theta ~ normal(theta_mean, 1);
  sigma_theta ~ normal(0,1);

  /* Calculate expectation for each event */
  for (i in 1:nobs) {
    real ps[nsamp];
    for (j in 1:nsamp) {
      real r; 
      real theta;
      r = r_thetas[i,j,1];
      theta = r_thetas[i,j,2];
      ps[j] = prior_conditioned_on_hyperprior(r, theta, mu_r, sigma_r, mu_theta, sigma_theta)*r;
    }
    target += log(sum(logps)) - log(nsamp*pi()/4);
  }
}
