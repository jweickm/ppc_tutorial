# -----------------------------------------------------------------------------
# Linear regression model
# Running both the prior predictive and the posterior predictive checks using Stan

# Jakob Weickmann, 2021
# jakob.weickmann@posteo.de
# in cooperation with Damian Bednarz and Lei Zhang
# damian.bednarz@posteo.de, lei-zhang.net

# Adapted from Richard McElreath, 2016 and the data set "Howell1"
# This data set is based on the data set from Nancy Howell accessible under
# https://tspace.library.utoronto.ca/handle/1807/18219

# -----------------------------------------------------------------------------
#### Construct Data #### 
# -----------------------------------------------------------------------------
# Load necessary packages
library(rstan) 
library(ggplot2)
library(bayesplot)
library(bayestestR)
library(rstudioapi)
library(ggpubr)
library(extraDistr)

# Set some options
# automatically saves a serialized version of the compiled model to the directory of the `.stan` file.
rstan_options(auto_write = TRUE)  

# detects the number of cores available and runs that many chains in parallel. This allows maximum performance, but the user may want to manually set the number of cores if not all cores should be used. 
options(mc.cores = parallel::detectCores()) 

ggplot2::theme_set(theme_classic())
# -----------------------------------------------------------------------------
# Loading the data
# -----------------------------------------------------------------------------
# Read in the Howell1 data set from the github repository of Richard McElreath
d <- read.csv("https://raw.githubusercontent.com/jweickm/ppc_tutorial/main/data/Howell1.csv", sep = ";")
str(d) 

df <- d[ d$age >=  18, ] # filter only for the adults
str(df)

custom.summary <- function(x, ...){
  c(mean   = mean(x, ...),
    sd     = sd(x, ...),
    median = median(x, ...),
    min    = min(x, ...),
    max    = max(x, ...), 
    n      = as.integer(length(x, ...)))
}

df_summary <- apply(df, 2, custom.summary)
mean_weight <- df_summary["mean", "weight"]

# define the values for the switch variable. This allows us to use the same Stan script for both the prior predictive checks and the posterior predictive checks
priorCheck <- 0 # 0: Prior predictive check
posteriorCheck <- 1 # 1: Posterior predictive check

# The linear regression model looks as follows: 
# height_predicted_i = alpha + beta * (weight_i - mean_weight) + epsilon_i

# Defining the prior distribution (for the regression parameters) based on the existing domain-specific knowledge  
prior_alpha <- c(165, 30) # sets the mean and standard deviation of the normal prior distribution for the intercept, where xi = mean(x) (alpha in the linear model), in cm
prior_beta  <- c(0, 4) # sets the mean and standard deviation of the half-normal prior distribution for the slope (beta), in cm/kg
prior_sigma <- c(0, 5) # sets the shape parameters for the half-cauchy prior distribution of the noise term (sigma)

# -----------------------------------------------------------------------------
# INITIALIZING STAN
# -----------------------------------------------------------------------------
# Set the stan file and define the MCMC parameters
modelFile <- "linear_regression.stan"
nIter     <- 2000
nChains   <- 4 
nWarmup   <- floor(nIter/2)
nThin     <- 1

N <- length(df$height)

dataList <- list(N = N, 
                weight = df$weight, 
                mean_weight = mean_weight,
                height = df$height, 
                prior_alpha = prior_alpha, 
                prior_beta = prior_beta, 
                prior_sigma = prior_sigma,
                analysis_step = -1)
Seed = 1450154627

# -----------------------------------------------------------------------------
#### Running Stan with prior predictive checks #### 
# -----------------------------------------------------------------------------
# Let Stan know that we want to conduct a prior predictive check
dataList$analysis_step <- priorCheck

cat("Estimating", modelFile, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")

fit_reg_pripc <- stan(modelFile,
                    model_name = "Linear Regression Height Prior Predictive Check",
                    data    = dataList,
                    chains  = nChains,
                    iter    = nIter,
                    warmup  = nWarmup,
                    thin    = nThin,
                    init    = "random",
                    seed    = Seed)

cat("Finishing", modelFile, "model simulation ... \n")
endTime = Sys.time(); print(endTime)
cat("It took", as.character.Date(endTime - startTime), "\n")

# -----------------------------------------------------------------------------
#### Running Stan with posterior predictive checks #### 
# -----------------------------------------------------------------------------
# Let Stan know that we want to include the likelihood to update the posterior
dataList$analysis_step <- posteriorCheck

cat("Estimating", modelFile, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")

fit_reg_postpc <- stan(modelFile, 
                  model_name = "Linear Regression Height Posterior Predictive Check",
                  data    = dataList, 
                  chains  = nChains,
                  iter    = nIter,
                  warmup  = nWarmup,
                  thin    = nThin,
                  init    = "random",
                  seed    = Seed)

cat("Finishing", modelFile, "model simulation ... \n")
endTime = Sys.time(); print(endTime)  
cat("It took",as.character.Date(endTime - startTime), "\n")

# -----------------------------------------------------------------------------
#### Visualizing prior predictive checks #### 
# -----------------------------------------------------------------------------

# extract samples from the generated stan model
height_bar <- rstan::extract(fit_reg_pripc, pars = "height_bar", permuted = TRUE)$height_bar
# ---------------------------------------------------------------

# Initial plot of the data
scatterplot <- ggplot2::ggplot(df, aes(weight, height)) + 
  ggplot2::geom_point() + 
  ggplot2::geom_smooth(method = 'lm', colour = "indianred3") + 
  ggplot2::labs(title = "Scatterplot of height ~ weight with linear regression line",
       x = "weight in kg", 
       y = "height in cm")

# Plot for alpha prior distribution
p_alpha <- ggplot2::ggplot() + 
  xlim(-10, 300) + 
  ggplot2::geom_function(fun = dnorm, args = list(mean = prior_alpha[1], sd = prior_alpha[2]), colour = "forestgreen", size=1) + 
  ggplot2::geom_vline(aes(xintercept = 272), linetype = 'dotted') + 
  ggplot2::annotate("text", x = 260, y = 0.007, label = "tallest person ever recorded (272 cm)", angle = 90) + 
  ggplot2::geom_vline(aes(xintercept = 0), linetype = 'dotted') + 
  ggplot2::geom_vline(aes(xintercept = prior_alpha[1]), alpha = 0.3) + 
  ggplot2::labs(x = "alpha in cm", 
       y = "density", 
       title = "Prior distribution for alpha (intercept at mean weight)", 
       subtitle = paste("alpha ~ normal(", as.character(prior_alpha[1]), ",", as.character(prior_alpha[2]), ")", sep = '')) 

# Plot for beta prior distribution 
p_beta <- ggplot2::ggplot() + 
  xlim(0, 20)+ 
  ggplot2::geom_function(fun = dnorm, args = list(mean = prior_beta[1], sd = prior_beta[2]), colour = "dodgerblue2", size = 1) + 
  ggplot2::labs(x = "beta in cm/kg", 
       y = "density", 
       title = "Prior distribution for beta (slope)", 
       subtitle = paste("beta ~ half-normal(", as.character(prior_beta[1]), ",", as.character(prior_beta[2]), ")", sep = ''))

# Plot for sigma prior distribution 
p_sigma <- ggplot2::ggplot() + 
  xlim(0, 50) + 
  ggplot2::geom_function(fun = extraDistr::dhcauchy, args = list(prior_sigma[2])) + 
  ggplot2::labs(x = "sigma in cm", 
       y = "density", 
       title = "Prior distribution for sigma (noise)", 
       subtitle = paste("sigma ~ half-cauchy(", as.character(prior_sigma[1]), ",", as.character(prior_sigma[2]), ")", sep = ''))

initial_plots <- ggpubr::ggarrange(scatterplot, p_alpha, p_beta, p_sigma, 
                           labels= c("a", "b", "c", "d"),
                           ncol = 2, nrow = 2)
initial_plots

# prior kernel dens
bayesplot::color_scheme_set("blue")
prior_kernel_dens <- bayesplot::ppc_dens_overlay(y = df$height, 
                                                 yrep = rstan::extract(fit_reg_pripc, pars = "height_bar")$height_bar[1:40, ], 
                                                 alpha = 0.5, size = .8) + 
  xlim(-250, 500) + 
  ggplot2::ylab("Density") + 
  ggplot2::theme(axis.text=element_text(size=14),
        axis.title=element_text(size=16),
        legend.text=element_text(size=20))

prior_kernel_dens

# -----------------------------------------------------------------------------
#### Visualizing the posterior predictive check #### 
# -----------------------------------------------------------------------------

# extract samples
height_bar <- rstan::extract(fit_reg_postpc, pars = "height_bar", permuted = TRUE)$height_bar
alpha_post <- rstan::extract(fit_reg_postpc, pars = "alpha", permuted = TRUE)$alpha
beta_post <- rstan::extract(fit_reg_postpc, pars = "beta", permuted = TRUE)$beta
sigma_post <- rstan::extract(fit_reg_postpc, pars = "sigma", permuted = TRUE)$sigma

# ---------------------------------------------
# Calculate credible intervals for parameters
ci_alpha_eti <- bayestestR::ci(alpha_post, method = "ETI")
ci_alpha_hdi <- bayestestR::ci(alpha_post, method = "HDI")

ci_beta_eti <- bayestestR::ci(beta_post, method = "ETI")
ci_beta_hdi <- bayestestR::ci(beta_post, method = "HDI")

ci_sigma_eti <- bayestestR::ci(sigma_post, method = "ETI")
ci_sigma_hdi <- bayestestR::ci(sigma_post, method = "HDI")
# ---------------------------------------------------------------

# posterior kernel dens
bayesplot::color_scheme_set("blue")
posterior_kernel_dens <- 
  bayesplot::ppc_dens_overlay(y = df$height, 
                              yrep = rstan::extract(fit_reg_postpc, 
                                                    pars = "height_bar")$height_bar[1:40, ], 
                              alpha = 0.5, size = .8) + 
  xlim(50, 250) + 
  ggplot2::ylab("Density") + 
  ggplot2::theme(axis.text=element_text(size=14),
        axis.title=element_text(size=16),
        legend.text=element_text(size=20)) + 
  ggplot2::labs(title = "Posterior kernel density")

posterior_kernel_dens


p_alpha_post_gg <- ggplot2::ggplot(as.data.frame(alpha_post), aes(alpha_post)) + 
  ggplot2::geom_density(colour = "forestgreen", size = 1) + 
  ggplot2::labs(title = "Posterior probability distribution of alpha",
       xlab = "alpha") 
  
p_beta_post_gg <- ggplot2::ggplot(as.data.frame(beta_post), aes(beta_post)) + 
  ggplot2::geom_density(colour = "dodgerblue2", size = 1) + 
  ggplot2::labs(title = "Posterior probability distribution of beta",
       xlab = "beta") 

p_sigma_post_gg <- ggplot2::ggplot(as.data.frame(sigma_post), aes(sigma_post)) + 
  ggplot2::geom_density(size = 1) + 
  ggplot2::labs(title = "Posterior probability distribution of sigma",
       xlab = "sigma") 

ggpubr::ggarrange(p_alpha_post_gg, p_beta_post_gg, p_sigma_post_gg, posterior_kernel_dens,
                           labels= c("a", "b", "c", "d"),
                           ncol = 2, nrow = 2)
