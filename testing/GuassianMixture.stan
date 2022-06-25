data {
  int n_events; /* Number of events*/
  int n_kernels; /* Number of GMM components per event */
  
  real weights[n_events, n_kernels]; /* Weights for each kernel of each event */
  vector[2] means[n_events, n_kernels]; /* Mean Vector for each kernel of each event */
  matrix[2,2] covs[n_events, n_kernels]; /* Covariance Matrix for each kernel of each event */
  
  
  /* Priors */
  vector[2] mu_lambda_mean
  matrix[2,2] mu_lambda_covariance /* Will probably just set this to be the identity matrix */
  vector[2] sigma_lambda_covariance
}

