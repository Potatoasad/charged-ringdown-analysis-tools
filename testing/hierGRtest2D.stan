data {
  int Nobs;

  int NC; /* Number of GMM components. */
  real weights[Nobs, NC];
  vector[2] means[Nobs, NC]; /* Mf, af */
  matrix[2,2] covs[Nobs, NC];
}

transformed data {
  real log_weights[Nobs, NC];
  matrix[2,2] chol_covs[Nobs, NC];

  for (i in 1:Nobs) {
    for (j in 1:NC) {
      log_weights[i,j] = log(weights[i,j]);
      chol_covs[i,j] = cholesky_decompose(covs[i,j]);
    }
  }
}


parameters {
  vector<lower=0,upper=1>[2] Mu;

  /* The cholesky factor of the correlation matrix. */
  cholesky_factor_corr[2] Lhat;

  /* Vector of standard deviation *scales* in the eigendirections of sigma. */
  vector<lower=0>[2] s;
}

transformed parameters {
  matrix[2,2] Sigma;
  matrix[2,2] L;

  L = diag_pre_multiply(s, Lhat);
  Sigma = multiply_lower_tri_self_transpose(L);
}

model {
  /* Hyperparameter priors */
  s ~ normal(0, 1);
  Lhat ~ lkj_corr_cholesky(2); /* p(correlation_matrix) ~ det(c_m)^(2-1) */

  /* Marginal log-likelihood.  We model the likelihood function as a Gaussian
     mixture.  Since the population model is Gaussian, too, we can marginalize
     each term of the mixture over the true x = (dM, dchi) by doing Gaussian integrals:

     p(x | Mu, Sigma, data) ~ sum_over_GMM(wt_GMM N[Mu,Sigma](x) N[x, Sigma_GMM](mu_GMM))

     Integrating over the (2D) x is just convolving the two Gaussians together, with result

     int_over_x(p(x | Mu, Sigma, data)) = sum_over_GMM(wt_GMM N[Mu, Sigma + Sigma_GMM](mu_GMM))

     (each term here is *also* the evidence, and factors into the generated quantities draw below).

     Since we can analytically marginalize over x, we don't need it as a
     parameter, and the sampling is very fast. */

  for (i in 1:Nobs) {
    real logp[NC];

    for (j in 1:NC) {
      logp[j] = multi_normal_lpdf(means[i,j] | Mu, covs[i,j] + Sigma) + log_weights[i,j];
    }

    target += log_sum_exp(logp);
  }
}

generated quantities {
  vector[2] xpop = multi_normal_rng(Mu, Sigma);
  real dMpop = xpop[1];
  real dchipop = xpop[2];

  vector[2] xtrue[Nobs];
  vector[Nobs] dMtrue;
  vector[Nobs] dchitrue;

  /* Sampling over xtrue is a bit more complicated.  First, we need to choose
     *which* KDE component we draw the xtrue sample from; we do this weighted by
     the evidence, as above.

     Once we have chosen a component, then the conditional for x is

     p(x | Mu, Sigma, data) ~ N[Mu, Sigma](x) N[x, Sigma_GMM](mu_GMM)

     which is *proportional* to a normal distribution with mean m given by

     m = (Sigma^-1 + Sigma_GMM^-1)^-1 (Sigma^-1 Mu + Sigma_GMM^-1 mu_GMM)
       = (I + Sigma Sigma_GMM^-1)^-1 (Mu + Sigma Sigma_GMM^-1 mu_GMM)

     and covariance S given by

     S = (Sigma^-1 + Sigma_GMM^-1)^-1
       = (I + Sigma Sigma_GMM^-1)^-1 Sigma

     In the second lines, we have written the formulas in a way that should be
     stable as long as Sigma_GMM is > Sigma (this is our expected regime).

      */

  for (i in 1:Nobs) {
    vector[NC] log_ev;
    int c;

    for (j in 1:NC) {
      log_ev[j] = multi_normal_lpdf(means[i,j] | Mu, Sigma + covs[i,j]) + log_weights[i,j];
    }

    c = categorical_logit_rng(log_ev);

    {
      matrix[2,2] I = [[1.0, 0.0], [0.0, 1.0]];
      matrix[2,2] SI = (I + Sigma / covs[i,c]);
      vector[2] m = SI \ (Mu + Sigma * (covs[i,c] \ means[i,c]));
      matrix[2,2] S = SI \ Sigma;

      xtrue[i] = multi_normal_rng(m, S);
      dMtrue[i] = xtrue[i][1];
      dchitrue[i] = xtrue[i][2];
    }
  }
}
