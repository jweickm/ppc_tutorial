// Linear regression model
// Jakob Weickmann, 2021
// jakob.weickmann@posteo.de

// in cooperation with Damian Bednarz and Lei Zhang
// damian.bednarz@posteo.de, lei-zhang.net
// ---------------------------------------------------------------------------------------
data {
  int<lower=0> N; // This is the number of data points, i.e. participants, in the case of prior predictive checks this encodes the number of simulations
  vector<lower=0>[N] weight; // In this regression, the weight is the independent variable. We are iterating on a model that allows us to predict the height from the weight in the data.
  real<lower=0> mean_weight;
  vector<lower=0>[N] height; // In the prior predictive check we are simulating the posterior distribution based only on the prior. The dependent variable `height` is only necessary when we include the likelihood function and conduct the posterior predictive check. 
  int<lower=0, upper=1> analysis_step; // boolean variable for the switch that allows us to use the same script with little variation depending on whether we are running the posterior (1) or the prior (0) predictive checks
  
  real prior_alpha[2]; // mean and standard deviation of the prior distribution from which we draw alpha (the intercept)
  real<lower=0> prior_beta[2]; // mean and standard deviation of the half-normal prior distribution from which we draw beta (the slope)
  real<lower=0> prior_sigma[2]; // shape parameters of the half-cauchy prior distribution (with a lower bound at 0) from which we draw sigma (the noise term)
} 

parameters {
  real alpha; // declaring the intercept
  real<lower=0> beta; // declaring the slope 
  real<lower=0> sigma; // declaring the standard deviation of the error term (cannot be negative)
} 

model {
  // priors
  alpha ~ normal(prior_alpha[1], prior_alpha[2]); // drawing the intercept
  beta  ~ normal(prior_beta[1], prior_beta[2]); // drawing the slope
  sigma ~ cauchy(prior_sigma[1], prior_sigma[2]); // drawing the noise term
  
  if (analysis_step == 1) {
    height ~ normal(alpha + beta * (weight - mean_weight), sigma); // this is the likelihood function that links the data (weight and height) with our priors. How likely is the data for a given parameter? P(y|Phi). Only active during the posterior predictive check and the only difference in the scripts function between prior predictive checks and posterior predictive checks. 
  }
}

generated quantities {
  vector[N] height_bar; // this is the modelled height
   // here we loop over all N to draw a value for the simulated height from our distribution 
   for (n in 1:N) {
      height_bar[n] = normal_rng(alpha + beta * (weight[n] - mean_weight), sigma); // calculate posterior distributions
  } 
}
