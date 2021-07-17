# -----------------------------------------------------------------------------
# Simple Reinforcement Learning Model 
# Running both the prior predictive and the posterior predictive checks using Stan

# Jakob Weickmann, 2021
# jakob.weickmann@posteo.de
# in cooperation with Damian Bednarz and Lei Zhang
# damian.bednarz@posteo.de, lei-zhang.net

# -----------------------------------------------------------------------------
#### Construct Data #### 
# -----------------------------------------------------------------------------
library(rstan)
library(ggplot2)
library(dplyr)
library(tidyr)
library(bayestestR)

# Set some options
# automatically saves a serialized version of the compiled model to the directory of the `.stan` file.
rstan_options(auto_write = TRUE)  

# detects the number of cores available and runs that many chains in parallel. This allows maximum performance, but the user may want to manually set the number of cores if not all cores should be used. 
options(mc.cores = parallel::detectCores()) 

# -----------------------------------------------------------------------------
# Loading the data
# -----------------------------------------------------------------------------

# The data are randomly generated. True parameters for the data generation are:
# lr  = rnorm(10, mean=0.6, sd=0.12); tau = rnorm(10, mean=1.5, sd=0.2)

# Read in the generated data from Github file
load(url('https://github.com/jweickm/ppc_tutorial/blob/main/data/rw.RData?raw=true'))
load("_data/rw_sim_data.RData")
sz <- dim(rw_mat)
nSubjects <- sz[1]
nTrials   <- sz[2]

# define the switch variable for the stan script
priorCheck <- 0 # 0: Prior predictive check
posteriorCheck <- 1 # 1: Posterior predictive check

# -----------------------------------------------------------------------------
# INITIALIZING STAN
# -----------------------------------------------------------------------------
# Set the stan file and define the MCMC parameters
modelFile <- "rescorla_wagner.stan"
nIter     <- 2000
nChains   <- 4 
nWarmup   <- floor(nIter/2)
nThin     <- 1

dataList <- list(nSubjects = nSubjects,
                 nTrials = nTrials, 
                 choice = rw_mat[, , 1],
                 reward = rw_mat[, , 2],
                 analysis_step = -1)
Seed = 1450154627

# -----------------------------------------------------------------------------
#### Running Stan & prior predictive checks #### 
# -----------------------------------------------------------------------------
dataList$analysis_step <- 0

cat("Estimating", modelFile, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")

fit_rw_pri <- stan(modelFile, 
               data    = dataList, 
               chains  = nChains,
               iter    = nIter,
               warmup  = nWarmup,
               thin    = nThin,
               init    = "random",
               seed    = Seed
)

cat("Finishing", modelFile, "model simulation ... \n")
endTime = Sys.time(); print(endTime)  
cat("It took",as.character.Date(endTime - startTime), "\n")

# -----------------------------------------------------------------------------
#### Running Stan & posterior predictive checks #### 
# -----------------------------------------------------------------------------
dataList$analysis_step <- 1

cat("Estimating", modelFile, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")

fit_rw_post <- stan(modelFile, 
               data    = dataList, 
               chains  = nChains,
               iter    = nIter,
               warmup  = nWarmup,
               thin    = nThin,
               init    = "random",
               seed    = Seed
)

cat("Finishing", modelFile, "model simulation ... \n")
endTime = Sys.time(); print(endTime)  
cat("It took",as.character.Date(endTime - startTime), "\n")

# -----------------------------------------------------------------------------
#### Model Summary and Diagnostics #### 
# -----------------------------------------------------------------------------
print(fit_rw_pri)
print(fit_rw_post)

plot_dens_lr_pri  <- stan_plot(fit_rw_pri, pars=c('lr_mu','lr'), show_density=T, fill_color = 'gray85')
plot_dens_tau_pri <- stan_plot(fit_rw_pri, pars=c('tau_mu','tau'), show_density=T, fill_color = 'gray85')
plot_dens_lr_post  <- stan_plot(fit_rw_post, pars=c('lr_mu','lr'), show_density=T, fill_color = 'skyblue')
plot_dens_tau_post <- stan_plot(fit_rw_post, pars=c('tau_mu','tau'), show_density=T, fill_color = 'skyblue')

ggpubr::ggarrange(plot_dens_lr_post, plot_dens_tau_post, 
                  labels= c("a", "b"),
                           ncol = 2, nrow = 1)

prior_comp_p <- ggpubr::ggarrange(norm_plot_dens_lr_pri, plot_dens_lr_pri, norm_plot_dens_tau_pri, plot_dens_tau_pri, 
                  labels = c("a. Normal(0,1)", "b. Cauchy(0,3)", "c. Normal(0,1)", "d. Cauchy(0,3)"),
                  font.label = list(size = 12),
                  hjust = -0.8,
                  ncol = 2, nrow = 2)
ggpubr::annotate_figure(prior_comp_p, 
                        top = "Comparison of Normal(0,1) (left) and Cauchy(0,3) (right) priors for the SD")


lr_post <- rstan::extract(fit_rw_post, pars = "lr_mu", permuted = TRUE)$lr_mu
tau_post <- rstan::extract(fit_rw_post, pars = "tau_mu", permuted = TRUE)$tau_mu

bayestestR::hdi(lr_post)
bayestestR::eti(lr_post)
bayestestR::hdi(tau_post)
bayestestR::eti(tau_post)

# -----------------------------------------------------------------------------
#### Visualizing prior predictive checks #### 
# -----------------------------------------------------------------------------

# extract samples from the generated stan model
y_pred_pri  <- rstan::extract(fit_rw_pri, pars ="y_pred", permuted = TRUE)$y_pred
y_pred_post <- rstan::extract(fit_rw_post, pars ="y_pred", permuted = TRUE)$y_pred

data_summary <- function(x) {
    m <- mean(x)
    ymin <- m-sd(x)
    ymax <- m+sd(x)
    return(c(y=m,ymin=ymin,ymax=ymax))
}

# -----------------------------------------------------------------------------
#### Violin plot of posterior means #### 
# -----------------------------------------------------------------------------
pars_value <- get_posterior_mean(fit_rw_post, pars=c('lr','tau'))[,5]
pars_name  <- as.factor(c(rep('lr',10),rep('tau',10)))
df <- data.frame(pars_value=pars_value, pars_name=pars_name)
  
myconfig <- theme_bw(base_size = 20) +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank() )

data_summary <- function(x) {
    m <- mean(x)
    ymin <- m-sd(x)
    ymax <- m+sd(x)
    return(c(y=m,ymin=ymin,ymax=ymax))
}

g1 <- ggplot(df, aes(x=pars_name, y=pars_value, color = pars_name, fill=pars_name)) 
g1 <- g1 + geom_violin(trim=TRUE, size=2)
g1 <- g1 + stat_summary(fun.data=data_summary, geom="pointrange", color="black", size=1.5)
g1 <- g1 + scale_fill_manual(values=c("#2179b5", "#c60256"))
g1 <- g1 + scale_color_manual(values=c("#2179b5", "#c60256"))
g1 <- g1 + myconfig + theme(legend.position="none")
g1 <- g1 + labs(x = '', y = 'parameter value') + ylim(0.3,2.2)
print(g1)



# make predictive checks
####################################################################
dL <- list(nSubjects=nSubjects,
                 nTrials=nTrials, 
                 choice= rw_mat[,,1], 
                 reward= rw_mat[,,2])

####################################################################
# overall mean of choosing the second (better) option
mean(dL$choice[,] == 2 )
mean(dL$reward[,] == 1 ) # overall mean of getting rewarded

# trial-by-trial sequence of choosing the 2nd option
y_mean = colMeans(dL$choice == 2)

y_pred_pri  <- rstan::extract(fit_rw_pri, pars ="y_pred", permuted = TRUE)$y_pred
y_pred_post <- rstan::extract(fit_rw_post, pars ="y_pred", permuted = TRUE)$y_pred

dim(y_pred_pri)
dim(y_pred_post)

# for prior predictive checks

y_pred_pri_mean_mcmc = apply(y_pred_pri==2, c(1,3), mean)
dim(y_pred_pri_mean_mcmc)  # [4000, 100]
y_pred_pri_mean = colMeans(y_pred_pri_mean_mcmc)
y_pred_pri_mean_hdi = apply(y_pred_pri_mean_mcmc, 2, HDIofMCMC)

#plot(1:100, colmeans(y_pred_pri_mean),type='b')

# =============================================================================
#### make plots #### 
# =============================================================================
myconfig <- theme_bw(base_size = 20) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank() )

df = data.frame(trial = 1:100,
                data  = y_mean,
                model = y_pred_pri_mean,
                hdi_l = y_pred_pri_mean_hdi[1,],
                hdi_h = y_pred_pri_mean_hdi[2,])

## time course of the choice
g2 = ggplot(df, aes(trial,data))
g2 = g2 + geom_line(size = 1.5, aes(color= 'data')) + geom_point(size = 2, shape = 21, fill='skyblue3',color= 'skyblue3')
#g2 = g2 + geom_ribbon(aes(ymin=hdi_l, ymax=hdi_h), linetype=2, alpha=0.3, fill = 'skyblue3')
g2 = g2 + geom_ribbon(aes(ymin=hdi_l, ymax=hdi_h, fill='model'), linetype=2, alpha=0.3)
g2 = g2 + myconfig + scale_fill_manual(name = '',  values=c("model" = "skyblue3")) +
  scale_color_manual(name = '',  values=c("data" = "skyblue"))  +
  labs(y = 'choosing correct (%)')
g2 = g2 + theme(axis.text   = element_text(size=22),
                axis.title  = element_text(size=25),
                legend.text = element_text(size=25))
g2
ggsave(plot = g2, "_plots/choice_seq_ppc.png", width = 8, height = 4, type = "cairo-png", units = "in")


## overall choice: true data (vertical line) + model prediction (hist)
tt_y = mean(df$data)
df2 = data.frame(model = rowMeans(y_pred_pri_mean_mcmc)) # overall mean, 4000 mcmc samples
g3 = ggplot(data=df2, aes(model)) + geom_histogram(binwidth =.005, alpha=.5, fill = 'skyblue3')
g3 = g3 + geom_vline(xintercept=tt_y, color = 'skyblue3',size=1.5)
g3 = g3 + labs(x = 'choosing correct (%)', y = 'frequency')
g3 = g3 + myconfig# + scale_x_continuous(breaks=c(tt_y), labels=c("event1")) 
g3 = g3 + theme(axis.text   = element_text(size=22),
                axis.title  = element_text(size=25),
                legend.text = element_text(size=25))
g3
ggsave(plot = g3, "_plots/choice_mean_ppc.png", width = 6, height = 4, type = "cairo-png", units = "in")



# for posterior predictive checks

y_pred_post_mean_mcmc = apply(y_pred_post==2, c(1,3), mean)
dim(y_pred_post_mean_mcmc)  # [4000, 100]
y_pred_post_mean = colMeans(y_pred_post_mean_mcmc)
y_pred_post_mean_hdi = apply(y_pred_post_mean_mcmc, 2, HDIofMCMC)

#plot(1:100, colmeans(y_pred_post_mean),type='b')

# =============================================================================
#### make plots #### 
# =============================================================================
myconfig <- theme_bw(base_size = 20) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank() )

df = data.frame(trial = 1:100,
                data  = y_mean,
                model = y_pred_post_mean,
                hdi_l = y_pred_post_mean_hdi[1,],
                hdi_h = y_pred_post_mean_hdi[2,])

## time course of the choice
g4 = ggplot(df, aes(trial,data))
g4 = g4 + geom_line(size = 1.5, aes(color= 'data')) + geom_point(size = 2, shape = 21, fill='skyblue3',color= 'skyblue3')
#g4 = g4 + geom_ribbon(aes(ymin=hdi_l, ymax=hdi_h), linetype=2, alpha=0.3, fill = 'skyblue3')
g4 = g4 + geom_ribbon(aes(ymin=hdi_l, ymax=hdi_h, fill='model'), linetype=2, alpha=0.3)
g4 = g4 + myconfig + scale_fill_manual(name = '',  values=c("model" = "skyblue3")) +
  scale_color_manual(name = '',  values=c("data" = "skyblue"))  +
  labs(y = 'choosing correct (%)')
g4 = g4 + theme(axis.text   = element_text(size=22),
                axis.title  = element_text(size=25),
                legend.text = element_text(size=25))
g4
ggsave(plot = g4, "_plots/choice_seq_ppc.png", width = 8, height = 4, type = "cairo-png", units = "in")


## overall choice: true data (vertical line) + model prediction (hist)
tt_y = mean(df$data)
df2 = data.frame(model = rowMeans(y_pred_post_mean_mcmc)) # overall mean, 4000 mcmc samples
g5 = ggplot(data=df2, aes(model)) + geom_histogram(binwidth =.005, alpha=.5, fill = 'skyblue3')
g5 = g5 + geom_vline(xintercept=tt_y, color = 'skyblue3',size=1.5)
g5 = g5 + labs(x = 'choosing correct (%)', y = 'frequency')
g5 = g5 + myconfig# + scale_x_continuous(breaks=c(tt_y), labels=c("event1")) 
g5 = g5 + theme(axis.text   = element_text(size=22),
                axis.title  = element_text(size=25),
                legend.text = element_text(size=25))
g5
ggsave(plot = g5, "_plots/choice_mean_ppc.png", width = 6, height = 4, type = "cairo-png", units = "in")