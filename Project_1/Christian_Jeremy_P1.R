###
#
# @author  Yuanming Shi and Christian McDaniel
# @version 1.0
# @date    Feb 5, 2018
# @see     LICENSE (MIT style license file)
#
#
#This script takes in a file as an argument at the command line and 
# performs multiple regression, and subsequent analysis and manipulations,
# on the data within. 
#
# > Rscript Christian_Jeremy_P1.R <filepath>
###

# read in filepath from the command line 
filepath = commandArgs(trailingOnly = TRUE)

# Problem 1-a
d <- read.csv(filepath)
fit <- lm(Y ~ . , d)  # fit the result Y with the rest of the data
summary(fit)          # give a summary of the fit
#plot(fit)            # plot the fitted results (not required this time)
deviance(fit)         # extract the deviance
sum(resid(fit)^2)     # compute the sum of squared error
anova(fit)            # compute the variance
dx <- d[1:15]         # slice the training data 
dy <- d$Y             # slice the prediction vector (Y)
t.test(dx, dy)        # t-test result
coef(summary(fit))    # extract the coefficients from the regression summary

# Problem 1-b
d$X42 <- dx$X4^2       # add a new transformed column into the old dataframe d
newFit <- lm(Y ~ ., d) 
summary(newFit)

# Problem 1-c
# trimmedX contains the columns remaining after removal of less important variables 
trimmedX <- c("X2", "X4", "X7", "X8", "X9", "X10", "X12", "X14", "X42", "Y")
trimmedX <- d[trimmedX]
fit_trimmed <- lm(Y ~ . , trimmedX)
summary(fit_trimmed)
deviance(fit_trimmed)

# Problem 1-d
d <- read.csv(filepath)
fit <- lm(Y ~ . , d)   # fit the result Y with the rest of the data
library(car)
vif(fit)