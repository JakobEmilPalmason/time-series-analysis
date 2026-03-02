
#----------------------------------------------------------------
# Example of maximum likelihood estimation of simple mean model
#----------------------------------------------------------------

# Load a few required packages
#install.packages("numDeriv")
#install.packages("animation")

library(numDeriv)
library(animation)


#----------------------------------------------------------------
# Simulate a sample from a normal distributed population
#----------------------------------------------------------------
# Specify the population parameters
mu <- 150
sigma <- 20
# Number of observations
n <- 100
# Randomly sample n observations the normal distribution
y <- rnorm(n = n, mean = mu, sd = sigma)

# Plot the distribution of the sample
hist(y, prob=TRUE, xlim=c(0,300))
xseq <- seq(50,250,len=1000)
# Plot the distribution of the population
lines(xseq, dnorm(xseq, mean = mu, sd = sigma))

#--------
# - Change the parameters mu and sigma, see how the values in the sample change (note the fixed xlim on the histotram)
# - Change the number of observations, when is the sample distribution closer to the population distribution?
#--------


#----------------------------------------------------------------
# Maximum-likelihood function
#----------------------------------------------------------------
# Calculate the maximum likelihood your self

# The normal pdf for the first observation is the likelihood for the first observation!
# We just use theta to make a bit more generic (now theta = c(mu,logsigma))
theta <- c(150, log(20))
# The likelihood (see description of function with ?dnorm)
dnorm(y[1], mean=theta[1], sd=exp(theta[2]))

# Plot the normal pdf and the first observation and its' likelihood value
xseq <- seq(-200, 400, len=1000)
val <- dnorm(y[1],mean=theta[1],sd=exp(theta[2]))
plot(xseq, dnorm(xseq, mean=theta[1], sd=exp(theta[2])), type="l", xlab="y", ylab="Normal pdf")
lines(c(y[1],y[1]), c(-1, val), col=2)
lines(c(-250,y[1]), c(val,val), col=2)
mtext("y[1]", 1, at=y[1], line=2, col=2)
mtext("L(y[1]|theta)", 2, at=val, line=2.5, col=2, padj=1)

# Finally, do this for all observations and sum: That's the likelihood of the sample given the parameter values in theta
prod(dnorm(y, mean=theta[1], sd=exp(theta[2])))

# Take the log don't give such small numbers
log(dnorm(y, mean=theta[1], sd=exp(theta[2])))
sum(log(dnorm(y, mean=theta[1], sd=exp(theta[2]))))

#--------
# - Go up and change theta, what is the likelihood? try some different values
# - What happens with the likelihood, does it always calculate nicely? where is it highest?
#--------


#----------------------------------------------------------------
# Negative log-likelihood is better for optimization
#----------------------------------------------------------------

# Take the negative to give as function to an optimizer
-sum(dnorm(y, mean=theta[1], sd=exp(theta[2]), log=TRUE))

#--------
# - This is the negative log-likelihood which is minimized by an optimizer
# - Its' minimum is the maximum of the likelihood 
#--------


#----------------------------------------------------------------
# Maximum-likelihood estimation
#----------------------------------------------------------------
# Run an optimization of the negative log-likelihood function

# Define the negative log-likelihood as a function of the parameters theta
neg_loglike <- function(theta) {
    # Take the mean
    mu <- theta[1]
    # Take the variance as the exponential (optimize -Inf to Inf, while variance goes from 0 to Inf)
    # Remember var = sigma^2
    sigma <- exp(theta[2])
    # Calculate the log to the probability function (pdf) for each observation and sum and make negative
    nll <- -sum(dnorm(y, mean=mu, sd=sigma, log=TRUE))
    # Another way: nll <- n*log(sigma) + 1/2 * sum((y - mu)^2/sigma^2)
    return(nll)
}

# Carry out the optimization
nlminb(start=c(0,0), objective=neg_loglike, control=list(trace=1))

#--------
# - Did it find estimates close to the population mu and sigma?
# - Should it have found the exact population mu and sigma!?
# - Have a look at the neg_loglike function to see how it takes the parameters in a vector
#--------


#----------------------------------------------------------------
# Generate an animation of the optimization
#----------------------------------------------------------------
# Optimise the negative loglikelihood function starting at (mu=0,sigma=0).
start <- c(0,0)
# nlminb outputs the current point at every iteration. The output is captured and saved in a temporary file.
ss <- capture.output(opt <- nlminb(start=start, objective=neg_loglike, control=list(trace=1)),
                     file = out.file <- tempfile())
                                        # Okay.. Load it here
trace <- read.table(out.file, stringsAsFactors=FALSE)[,-1]
colnames(trace) <- c("ll","mu","lvar")
trace$ll <- as.double(substr(trace$ll, 1L, 9L))
ymax <- dnorm(trace$mu, mean=trace$mu, sd = exp(trace$lvar))[nrow(trace)] * 1.2
# Animation running the optimisation
oopt <- ani.options(interval = .1)
xseq <- seq(-200,400,length.out=400)
opar <- par(mfrow=c(1,2), oma=c(3,3,4,0), mar=c(4,3,1,1), las=1, cex=0.8)
for (i in seq_len(nrow(trace))) {
    # The population distribution
    pdf <- dnorm(x=xseq, mean=trace$mu[i], sd=exp(trace$lvar[i]))
    #
    plot(xseq, pdf, type="l", xlab = "Observations", ylab = "PDF", ylim = c(0, ymax))
    legend("topright", lty=1, col=c(1,4,2), c("Estimated","Population","Observations"))
    lines(xseq, dnorm(x=xseq, mean=mu, sd=sigma), col="blue")
    rug(y, col="red", lwd=1)
    #  
    plot(seq_len(i), trace$ll[1:i], xlim = c(1,nrow(trace)), ylim = range(trace$ll), type="p", 
         xlab = "Iteration", log="y", pch=16)
    #  
    title(main = bquote(atop(list(Iteration == .(i), 
                                  -log(L) == .(formatC(trace$ll[i], digits=1, format="f"))
                                  ), 
                             list(mu    == .(formatC( trace$mu[i], digits=1, format="f")),
                                  sigma == .(formatC( sqrt(exp(trace$lvar[i])), digits=1, format="f"))
                                  )
                             )
                        ), outer = TRUE)
    #
    ani.pause(0.8)
}

# Restore the plotting pars.
par(opar)

#--------
# - Try going up all the way up and increase sigma to make the observations more spread on the axis (red lines)
#--------

#----------------------------------------------------------------
# Generate another animation of the optimization
#----------------------------------------------------------------
# Contour plot of the neg_loglikelihood function of the parameter space
mu_seq <- seq(-20,300, length.out=100)
lsd <- seq(-3, 12, length.out=100)
#
grid <- as.matrix(expand.grid(mu = mu_seq, lsd = lsd))
#
llc <- numeric(nrow(grid))
#
for (i in seq_len(nrow(grid))) {
    llc[i] <- neg_loglike(grid[i,])
}
#
for (i in seq_len(nrow(trace))) {
    contour(x = mu_seq,
            y = lsd,
            z = matrix(log(llc),
                       ncol=length(mu_seq)), 
            levels = c(seq(log(min(llc)), 8, length.out=20),
                       seq(8,20,length.out=30)),
            xlab = bquote(mu_seq),
            ylab = bquote(log(sigma^2)))
    #  
    points(mu,log(sigma^2), col="blue", pch=16)
    #  
    lines(trace$mu[seq_len(i)], trace$lvar[seq_len(i)], col="red",lwd=2)
    points(trace$mu[seq_len(i)], trace$lvar[seq_len(i)], col="red",pch=16)
    #  
    title(main = bquote(atop(list(Iteration == .(i), 
                                  -log(L) == .(formatC(trace$ll[i], digits=1, format="f"))
                                  ), 
                             list(mu    == .(formatC( trace$mu[i], digits=1, format="f")),
                                  sigma == .(formatC( sqrt(exp(trace$lvar[i])), digits=1, format="f"))
                                  )
                             )
                        ))
    #  
    ani.pause(0.4)
}

#--------
# - Dwell a bit over how the optimizer take steps towards the minimum
#--------



#----------------------------------------------------------------
# The Maximum Likelihood Estimated (MLE) values and compare to regression
#----------------------------------------------------------------

# The population mean
mu
# The MLE mean
opt$par[1]

# The population standard deviation
sigma
# The MLE standard deviation
sqrt(exp(opt$par[2]))

# Use linear regression and compare
fit <- lm(y ~ 1)
summary(fit)
# The sigma is estimated from the residuals
sd(fit$residuals)

#--------
# - Are the estimates with the two methods exactly the same?
# - Note the exponential for the standard deviation. Because of the optimizer, the exponential was used such that the optimizer can work in the range from -Inf to Inf, and the value used for sigma^2 goes from 0 to Inf (the variance cannot be negative). If not used the optimizer will often crash.
#--------


#----------------------------------------------------------------
# What about the estimated uncertainty of the parameters with MLE?
#----------------------------------------------------------------
# I.e. the standard error ("Std. Error" in regression summary)
summary(fit)

# Approximate the Hessian
H <- hessian(neg_loglike, opt$par)

# Fisher's information matrix
I <- qr.solve(H)

# Covariance matrix of the estimates
cov2cor(I)

# Standard error (Standard deviation estimate of the theta parameters, so mu and exp(sigma^2))
sqrt(diag(I))