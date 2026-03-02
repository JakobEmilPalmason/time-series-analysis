
# Link to send out
# https://forms.gle/XKGfw1tNyjp6263d6

# First time you need to install the package gsheet:
#install.packages("gsheet")

# Load the package
library(gsheet)

################################
# Open data from a google sheet and process a bit
D <- gsheet2tbl("https://docs.google.com/spreadsheets/d/1pzM95iCcYmgmD1pEdcTmY6m6H6IoIg2I7qNArhGideA/edit?usp=sharing", sheetid = NULL)
attr(D, "spec") <- NULL
D <- as.data.frame(D)
D[ ,1] <- as.POSIXct(D[ ,1], format="%d/%m/%Y %H:%M:%S", tz="GMT")
D <- D[complete.cases(D), ]
# Give new names to the columns
names(D)[1] <- "time"
names(D)[2] <- "weight"
names(D)[3] <- "height"
names(D)[4] <- "foot"
names(D)[5] <- "nose"
names(D)[6] <- "ear"
names(D)[7] <- "gender"


# Take data from today only
D <- D[D$time > as.POSIDct("2022-01-14"), ]
# 
str(D)
D
summary(D)
# How are they distributed, any ourliers?
boxplot(D[ ,2:3])
boxplot(D[ ,4:6])
# See if we can easily see dependence
pairs(D[ ,c("weight","height","foot","nose","ear")], panel=panel.smooth)

################################################################
# Are we qualified to remove some extreme values!?
D <- D[20 < D$foot & D$foot < 32, ]
D <- D[D$height < 210, ]
D <- D[D$nose < 7, ]
D <- D[D$ear < 9, ]


################################################################
# Let's make a GLM for predicting the height

# Output variable
y <- D$height

# First model, only an intercept, hence the design matrix
X <- cbind(1, D$weight)

# The parameter estimates
thetahat1 <- solve(t(X) %*% X) %*% t(X) %*% y

thetahat1

# Now let's investigate the residuals
D$resid1 <- y - X %*% thetahat1

# See if we can easily see dependence to the residuals
pairs(D[ ,c("resid1","height","weight","foot","nose","ear")], panel=panel.smooth)


# Add more variables?


# Use lm for the same
fit <- lm(height ~ weight, D)
summary(fit)


################################################################
# Now let's do recursive updating using RLS

# Set the initial value of R and theta
p <- ncol(X)
N <- nrow(X)
R <- diag(1E-8, p)
theta <- c(150,0)#rep(0, p)
Theta <- matrix(NA, nrow=N, ncol=p)

# Iterate through and estimate the parameters
for(i in 1:N){
  (x <- X[i, ])
  # Update
  (R <- R + x %*% t(x))
  (theta <- theta + solve(R) %*% x %*% (y[i] - t(x) %*% theta))
  Theta[i, ] <- theta
}

Theta
thetahat1

par(mfrow=c(2,1))
plot(Theta[ ,1])
plot(Theta[ ,2])

