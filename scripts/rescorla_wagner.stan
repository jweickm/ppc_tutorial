// Modelling reinforcement learning in a two-armed bandit task with a Rescorla-Wagner model
// Jakob Weickmann, 2021
// jakob.weickmann@posteo.de

// in cooperation with Damian Bednarz and Lei Zhang
// damian.bednarz@posteo.de, lei-zhang.net
// ---------------------------------------------------------------------------------------

data {
  int<lower=1> nSubjects; // This is the number of participants, in the case of prior predictive checks this encodes the number of simulations. There has to be at least 1 participant
  int<lower=1> nTrials; // This is the number of trials per participant, at least 1 trial
  int<lower=1,upper=2> choice[nSubjects, nTrials]; // This is the choice that was made by a participant in a given trial, either 1 or 2
  real<lower=-1, upper=1> reward[nSubjects, nTrials]; // This variable encodes whether a reward was received in a given trial, 1 if a reward was received, and -1 if no reward was received
  int<lower=0, upper=1> analysis_step; // This is the boolean variable for the switch that allows us to use the same script with little variation depending on whether we are running the posterior (1) or the prior (0) predictive checks
}

transformed data {
  vector[2] initV;  // Declaring the variable initial values for V (the value attached to each choice)
  initV = rep_vector(0.0, 2); // V is a vector with two entries, one for each choice. V is initiated with the precursor variable `initV`, that is initialized with two zeros.
}

parameters {
  // group-level parameters
  real lr_mu_raw;  // the mean of the distribution from which we draw the learning rate (lr), assuming that the learning rates of the subjects are drawn from a common distribution
  real<lower=0> lr_sd_raw; // the standard deviation for the above distribution. Must be positive
  
  real tau_mu_raw; // same as above for tau, the 'temperature parameter'. The higher the value of tau, the higher the probability to choose the option with the higher value associated to it (especially for small value differences)
  real<lower=0> tau_sd_raw; // the standard deviation for the distribution, from which we draw tau
  
  // subject-level raw parameters
  vector[nSubjects] lr_raw; // declaring the subject-specific value for the individual learning rate
  vector[nSubjects] tau_raw; // declaring the subject-specific value for the 'temperature parameter' tau
}

transformed parameters {
  vector<lower=0,upper=1>[nSubjects] lr; // defining the constraints for the per subject learning rate parameter (lr)
  vector<lower=0,upper=50>[nSubjects] tau; // defining the constraints for the per subject temperature parameter (tau), 5 is chosen here as it appears to yield usable results in practice
  
  for (s in 1:nSubjects) {
    lr[s]  = Phi_approx( lr_mu_raw  + lr_sd_raw * lr_raw[s] ); // the function Phi_approx maps a distribution on the [0,1] parameter space. T
    tau[s] = Phi_approx( tau_mu_raw + tau_sd_raw * tau_raw[s] ) * 50; // the same is true for the temperature parameter tau. It is instead mapped to a [0,5] parameter space. 
  }
}

model {
  lr_mu_raw  ~ normal(0,1);
  tau_mu_raw ~ normal(0,1);
  lr_sd_raw  ~ normal(0,1);
  tau_sd_raw ~ normal(0,1);
  
  lr_raw  ~ normal(0,1);
  tau_raw ~ normal(0,1);
  
  // Only active during the posterior predictive check and the only difference in the scripts function between prior predictive checks and posterior predictive checks. 
  if (analysis_step == 1) {
    for (s in 1:nSubjects) { // loops over all subjects
      vector[2] v; // declare the variable value (v)
      real pe; // declare the variable prediction error (pe)
      v = initV; // set the value to the initial value (0,0) for that subject

      for (t in 1:nTrials) { // loops over all trials for that subject
        choice[s,t] ~ categorical_logit( tau[s] * v ); // this is the likelihood function that links the data (choice and reward) with our priors. How likely is the data (choice) for a given parameter (tau, v)? The `categorical_logit` function applies the softmax function internally to convert a vector (tau[s] * v) to a simplex.
        pe = reward[s,t] - v[choice[s,t]]; // updating the prediction error depending on the outcome of the trial and the expected reward
        v[choice[s,t]] = v[choice[s,t]] + lr[s] * pe; // updating the value of each choice depending on the prediction error, the learning rate and the value previously attached to that choice
        // the combination of the above statements is the Rescorla Wagner Model expressed in probability statements
      }
    }
  }
}

generated quantities {
  real<lower=0,upper=1> lr_mu; // declaring the mean of the learning rate
  real<lower=0,upper=50> tau_mu; // declaring the mean of tau
  
  int y_pred[nSubjects, nTrials]; // declaring the modelled choice (predicted)
  
  lr_mu  = Phi_approx(lr_mu_raw);
  tau_mu = Phi_approx(tau_mu_raw) * 50;
  y_pred = rep_array(-999,nSubjects ,nTrials);
  
  { // local block
    for (s in 1:nSubjects) {
      vector[2] v; 
      real pe;    
      v = initV;
      for (t in 1:nTrials) {        
        y_pred[s,t] = categorical_logit_rng( tau[s] * v ); // draws one tau value from the tau distribution for this iteration of the MCMC
        pe = reward[s,t] - v[choice[s,t]]; // prediction error according to the reward and choice from the data
        v[choice[s,t]] = v[choice[s,t]] + lr[s] * pe; // choice update according to the data 
      }
    }    
  }  
}
