
# Example dataset
D <- read.table("../../04-data/co2_mm_mlo.csv", skip=40, sep=",", header=TRUE)
D <- D[ ,c("decimal.date","average")]
names(D) <- c("time","co2")

# Plot
plot(D$time, D$co2, pch=19, cex=0.7)

# Fit global linear trend model
fit <- lm(co2 ~ time, D)

# Plot the data and the regression line
plot(D$time, D$co2, pch=19, cex=0.7)
abline(fit, col=2)

# Fit global linear trend model, with second order polynomial
D$timesq <- D$time^2
fit2 <- lm(co2 ~ time + timesq, D)

# Plot the data and the regression line
plot(D$time, D$co2, pch=19, cex=0.7)
lines(D$time, fit2$fitted.values, type="l", col=2)

D$time3 <- D$time^3
fit3 <- lm(co2 ~ time + timesq + time3, D)

# Plot the data and the regression line
plot(D$time, D$co2, pch=19, cex=0.7)
lines(D$time, fit3$fitted.values, type="l", col=2)


# Fit global linear trend model, add seasonal
D$sin1 <- sin(2*pi*D$time/1)
D$cos1 <- cos(2*pi*D$time/1)
fit3 <- lm(co2 ~ time + timesq + sin1 + cos1, D)

# Plot the data and the regression line
plot(D$time, D$co2, pch=19, cex=0.7)
lines(D$time, fit3$fitted.values, type="l", col=2)

plot(fit3$residuals)
acf(fit3$residuals)


# Fit global linear trend model, add seasonal
D$sin2 <- sin(2*pi*2*D$time/1)
D$cos2 <- cos(2*pi*2*D$time/1)
fit4 <- lm(co2 ~ time + timesq + sin1 + cos1 + sin2 + cos2, D)
summary(fit4)

# Plot the data and the regression line
plot(D$time, D$co2, pch=19, cex=0.7)
lines(D$time, fit4$fitted.values, type="l", col=2, lwd=2)

plot(fit4$residuals)
acf(fit4$residuals)


## ... make predictions, well, gotta' be careful!...

