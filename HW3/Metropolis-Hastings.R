library(mvtnorm)
library(methods)

normal_proposal <- function(x, sigma){
  return(mvrnorm(1, mu=rep(0, length(x)), Sigma=2.4^2*sigma/length(x)))
}

normal_proposal_prob <- function(x, sigma){
  return(dmvnorm(x=x, mean=rep(0, length(x)), sigma=2.4^2*sigma/length(x)))
}

bivariate_target <- function(x){
  return(dmvnorm(x=x, mean=rep(0, length(x)), sigma=diag(nrow=length(x))))
}

log_prob <- function(x1, x0){
  return(log(bivariate_target(x1)) + log(normal_proposal_prob(x0, mhs_sigma)))
}

MetropolisHastings <- function(x0, target, proposal, niter=1000){
    mhs_target <- target
    mhs_proposal <- proposal
    mhs_proposal_prob <- proposal_prob
    # fine-tune variance
    # run algorithm
    mhs_sigma <- diag(nrow=length(x0))
    mysamps = matrix(NA, nrow=niter, ncol=length(x0))
    for (i in 1:niter){
      print(i)
      x1 <- proposal(x0, sigma)
      logr <- log_prob(x1, x0) - log_prob(x0, x) 
      print(logr)
      if (log(runif(1)) < logr) x0 <- x1
      mysamps[i,] <- x0
    }
    return(mysamps)
}

bivariate_start <- c(0,0)

MetropolisHastings(bivariate_start, bivariate_target, normal_proposal)
