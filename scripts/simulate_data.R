# -----------------------------------------------------------------------------
# Simulate Rescorla-Wagner prediction error learning data for a two-armed bandit task

# Jakob Weickmann, 2021
# jakob.weickmann@posteo.de

# Based on generative code by Nathaniel Phillips
# https://rpubs.com/YaRrr/ReinforcementLearningSimulation
# -----------------------------------------------------------------------------

# lr  = rnorm(10, mean=0.6, sd=0.12); tau = rnorm(10, mean=1.5, sd=0.2)
nTrials <- 100
# These functions define the main learning and choice processes. The learning function rw.fun() is the Rescorla-Wagner prediction error learning rule. The choice function softmax.fun() is softmax.
# Rescorla-Wagner prediction error updating function
rw.fun <- function(exp.prior = c(1, 1),      # A vector of prior expectations
                   new.inf = c(NA, NA),        # A vector of new information (NAs except for selected option)
                   lr = rnorm(1, mean=0.6, sd=0.12)) {   # Updating rate
  
  # Save new expectations as prior
  exp.new <- exp.prior
  
  # Determine which option was selected
  selection <- which(is.finite(new.inf))
  
  # Update expectation of selected option
  exp.new[selection] <- exp.prior[selection] + lr * (new.inf[selection] - exp.prior[selection])
  
  return(exp.new)
}

# Softmax selection function
softmax.fun <- function(exp.current = c(1, 1),
                        tau = rnorm(1, mean=1.5, sd=0.2)
) {
  
  output <- exp(exp.current * tau) / sum(exp(exp.current * tau))
  
  return(output)
}

## The main simulation funciton is rl.sim.fun(). The function returns a dataframe containing the main results of the agent (and plots the agent’s cumulative earnings when plot = TRUE)

# Create main simulation function
rl.sim.fun <- function(n.trials = nTrials,     # Trials in game
                       option.probability = c(0.25, 0.75), # Probability for each option to produce a reward
                       n.options = 2, 
                       prior.exp.start = rep(0, 2), 
                       tau = rnorm(1, mean=1.5, sd=0.2), 
                       lr = rnorm(1, mean=0.6, sd=0.12)) {
  
  # Get some game parameters from inputs
  n.options <- length(option.probability)
  
  # Create outcome matrix giving the outcome on each trial for each option
  outcome.mtx <- matrix(NA, nrow = n.trials, ncol = n.options)
  
  reward_seq <- rbinom(n.trials, 1, option.probability[2]) + 1
  
  for(option.i in 1:n.options) {
    outcome.mtx[,option.i] <- reward_seq == option.i
  }
  
  outcome.mtx[outcome.mtx == FALSE] <- -1
  outcome.mtx[outcome.mtx == TRUE] <- 1
  
  # Create exp.prior and exp.new matrices
  #  These will hold the agent's expectation (either prior or new) 
  #   of each option on each trial
  
  exp.prior.mtx <- matrix(NA, nrow = n.trials, ncol = n.options)
  exp.prior.mtx[1,] <- prior.exp.start
  exp.new.mtx <- matrix(NA, nrow = n.trials, ncol = n.options)
  
  # Now create some matrices to store values
  
  selection.v <- rep(NA, n.trials)      # Actual selections
  outcome.v <- rep(NA, n.trials)        # Actual outcomes
  selprob.mtx <- matrix(NA,             # Selection probabilities
                        nrow = n.trials,
                        ncol = n.options)
  
  # RUN SIMULATION!
  
  for(trial.i in 1:n.trials) {
    
    # STEP 0: Get prior expectations for current trial
    exp.prior.i <- exp.prior.mtx[trial.i,]
    
    # STEP 1: SELECT AN OPTION
    # Selection probabilities
    selprob.i <- softmax.fun(exp.current = exp.prior.i, 
                             tau = tau)
    
    # Select an option
    selection.i <- sample(1:n.options, size = 1, prob = selprob.i)
    
    # Get outcome from selected option
    outcome.i <- outcome.mtx[trial.i, selection.i]
    
    # STEP 3: CREATE NEW EXPECTANCIES
    # Create a new.inf vector with NAs except for outcome of selected option
    new.inf <- rep(NA, n.options)
    new.inf[selection.i] <- outcome.i
    
    # Get new expectancies
    new.exp.i <- rw.fun(exp.prior = exp.prior.i,
                        new.inf = new.inf,
                        lr = lr)
    
    # assign new expectatations to exp.new.mtx[trial.i,]
    # and prior.expecation.mtx[trial.i + 1,]
    exp.new.mtx[trial.i,] <- new.exp.i
    
    if(trial.i < n.trials) {
      exp.prior.mtx[trial.i + 1,] <- new.exp.i
    }
    
    # Save some values
    selprob.mtx[trial.i,] <- selprob.i  # Selection probabilities
    selection.v[trial.i] <- selection.i # Actual selection
    outcome.v[trial.i] <- outcome.i     # Actual outcome
  }
  
  # Put main results in a single dataframe called sim.result.df
  sim.result.df <- data.frame("selection" = selection.v, 
                              "outcome" = outcome.v,
                              "outcome.cum" = cumsum(outcome.v),
                              stringsAsFactors = FALSE)
  
  # Now return the main simulation dataframe
  return(sim.result.df)
}


# Simulate the data for 10 subjects
set.seed(100) # For replicability
nSubjects <- 10
rw_mat <- array(NA, c(nSubjects, nTrials, 3))
for (subject in 1:nSubjects){
  trial_mat <- rl.sim.fun()
  rw_mat[subject, , 1] <- trial_mat[, 1]
  rw_mat[subject, , 2] <- trial_mat[, 2]
  rw_mat[subject, , 3] <- trial_mat[, 3]
}

save(rw_mat, file = "_data/rw_sim_data.RData")
