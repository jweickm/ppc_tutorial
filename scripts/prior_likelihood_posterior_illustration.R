library(ggplot2)
library(dplyr)
library(tidyr)
library(latex2exp)
library(ggpubr)

ggplot2::theme_set(theme_classic2())

# this is a grid approximation with 1000 parts
theta <- seq(.0005, .9995, .001)
res <- 1/length(theta)

weak_prior <- dbeta(x = theta, shape1 = 5, shape2 = 5) * res
strong_prior <- dbeta(x = theta, shape1 = 25, shape2 = 25) * res
flat_prior <- dunif(x = theta, min = 0, max = 1) * res
weak_prior_b <- dbeta(x = theta, shape1 = 2, shape2 = 8) * res
# ---------------------------------------
# ---------------------------------------
# model 100 throws with a flat, 2 weak and a strong prior. posterior update for each throw
observations <- rbinom(80, 1, 0.6)
head_tails <- c("T", "H")
all_throws <- head_tails[observations+1]
prior_1 <- flat_prior
prior_2 <- weak_prior
prior_2b <- weak_prior_b
prior_3 <- strong_prior
par(mar = c(2.5, 2, 2.5, 0.5))
layout(matrix(c(1,2,3,4,17, 5,6,7,8,17, 9,10,11,12,17, 13,14,15,16,17), ncol=4), heights=c(2,2,2,2,1), widths = c(.8, 1, 1, 1))

plot.new()
text(0.5, 0.5, paste("a) Flat prior", "~ Uniform(0,1)", sep="\n"), cex=1.5, font=1)
plot.new()
text(0.5, 0.5, paste("b) Weak prior A", "~ Beta(5,5)", sep="\n"), cex=1.5, font=1)
plot.new()
text(0.5, 0.5, paste("c) Weak prior B", "~ Beta(2,8)", sep="\n"), cex=1.5, font=1)
plot.new()
text(0.5, 0.5, paste("d) Strong prior", "~ Beta(25,25)", sep="\n"), cex=1.5, font=1)

for (i in 1:length(observations)){
  likelihood <- dbinom(x = observations[i], size = 1, prob = theta)
  # flat prior
  posterior_1 <- likelihood * prior_1
  posterior_1 <- posterior_1 / sum(posterior_1)
  # weak prior
  posterior_2 <- likelihood * prior_2
  posterior_2 <- posterior_2 / sum(posterior_2)
  # weak prior 2
  posterior_2b <- likelihood * prior_2b
  posterior_2b <- posterior_2b / sum(posterior_2b)
  # strong prior
  posterior_3 <- likelihood * prior_3
  posterior_3 <- posterior_3 / sum(posterior_3)
  
  # likelihood <- likelihood / sum(likelihood)
  throw <- head_tails[observations[i] + 1]

  if (is.element(i, c(5, 20, 80))){
    # flat prior
    plot(x = theta, y = posterior_1, 
         type = "l", 
         ylim=c(0, 0.01),
         yaxt='n', 
         ylab="",
         xlab = "",
         panel.first = c(abline(h=seq(0, 0.01, 0.002), col="gray94"), abline(v=seq(0, 1, 0.1), col="gray94")))
    lines(x = theta, y = flat_prior, type = "l", lty=3)
    mtext(paste("N =", i), side=3, line=.5, adj=0.5)
    text(0.15, 0.0085, paste("H = ", sum(observations[1:i])))
      
    if (i == 5){
      title(ylab = "Plausibility", line = .5)
    }
    
    # weak prior 1
    plot(x = theta, y = posterior_2, 
         type = "l", 
         ylim=c(0, 0.01),
         yaxt='n', 
         ylab="",
         xlab = "",
         panel.first = c(abline(h=seq(0, 0.01, 0.002), col="gray94"), abline(v=seq(0, 1, 0.1), col="gray94")))
    lines(x = theta, y = weak_prior, type = "l", lty=3)
    mtext(paste("N =", i), side=3, line=.5, adj=0.5)
    text(0.15, 0.0085, paste("H = ", sum(observations[1:i])))
    
    if (i == 5){
      title(ylab = "Plausibility", line =.5)
    }
    
    # weak prior 2
    plot(x = theta, y = posterior_2b, 
         type = "l", 
         ylim=c(0, 0.01),
         yaxt='n', 
         ylab="",
         xlab = "",
         panel.first = c(abline(h=seq(0, 0.01, 0.002), col="gray94"), abline(v=seq(0, 1, 0.1), col="gray94")))
    lines(x = theta, y = weak_prior_b, type = "l", lty=3)
    mtext(paste("N =", i), side=3, line=.5, adj=0.5)
    text(0.15, 0.0085, paste("H = ", sum(observations[1:i])))
    
    if (i == 5){
      title(ylab = "Plausibility", line =.5)
    }
    
    # strong prior
    plot(x = theta, y = posterior_3, 
         type = "l", 
         ylim=c(0, 0.01),
         yaxt='n', 
         ylab="",
         xlab = "",
         panel.first = c(abline(h=seq(0, 0.01, 0.002), col="gray94"), abline(v=seq(0, 1, 0.1), col="gray94")))
    lines(x = theta, y = strong_prior, type = "l", lty=3)
    mtext(paste("N =", i), side=3, line=.5, adj=0.5)
    text(0.15, 0.0085, paste("H = ", sum(observations[1:i])))
    title(xlab = TeX("$\\theta$"), line=1.5)
  }
  
    if (i == 5){
      title(ylab = "Plausibility", line =.5)
    }
  
  # lines(x = theta, y = likelihood, type = "l", lty=1, col = 'red')
  prior_1 <- posterior_1
  prior_2 <- posterior_2
  prior_2b <- posterior_2b
  prior_3 <- posterior_3
}
plot(0,0, type='l', bty='n', xaxt='n', yaxt='n')
legend("topright", legend=c("Prior", "Posterior"), lty=c(3,1))
