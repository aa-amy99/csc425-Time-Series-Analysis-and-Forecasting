
#s&P

library(tseries)
library(fBasics)
library(zoo)
library(forecast)
library(lmtest)
library(fUnitRoots)
library(rugarch) 

setwd("C:/Users/sbajaj4/Desktop")

myd=read.table("S&P_5yrs.csv", header=T, sep=',')

head(myd)

myd[,2]
price=myd[,6]
plot(price, main="Plot of Adjusted Price", type="l")

#create ts object
ats=ts(price, start=c(2013, 2), frequency = 252)
head(ats)
plot(ats)
plot(ats, main="Adjusted Closing Price", type="l")

par(mfcol=c(1,1))


#Checking outliers
tsoutliers(ats)

#removing them
atsclean=tsclean(ats)
plot(atsclean)
plot(ats, type='l')

#missing values
ats=na.approx(ats)
plot(ats)



#Analysis of distribution
basicStats(price)
# NORMALITY TESTS 
# Perform Jarque-Bera normality test. 
normalTest(ats,method=c("jb")) 

par(mfcol=c(1,1))
hist(ats, xlab="Daily Price", prob=TRUE, main="Distribution of S&P Daily Price")
xfit<-seq(min(ats),max(ats),length=40)
yfit<-dnorm(xfit,mean=mean(ats),sd=sd(ats))
lines(xfit, yfit, col="blue", lwd=2)

# CREATE NORMAL PROBABILITY PLOT
qqnorm(price)
qqline(price, col = 2)


# ACF ANALYSIS
acf(coredata(ats),plot=T, lag=20, main="ACF Plot of S&P Prices")

#Pacf
pacf(coredata(ats), main="Pacf Plot")

#Ljung Box Test
Box.test(ats,lag=3,type='Ljung-Box')
Box.test(ats,lag=7,type='Ljung-Box')

# Dickey Fuller test
# tests for AR model with time trend
adfTest(ats, lags=3, type=c("ct"))
adfTest(ats, lags=7, type=c("ct"))

# tests for AR model with no time trend
adfTest(price, lags=3, type=c("c"))
adfTest(price, lags=7, type=c("c"))


# APPLYING DIFFERENCING TO DATA
dx=diff(ats)
acf(coredata(dx), plot=T, lag=40)
pacf(coredata(dx), plot=T, lag=40)

#check seasonal diferencing
sdx=diff(dx,6)
acf(coredata(sdx), plot=T, lag=40)
pacf(coredata(sdx), plot=T, lag=40)



# Dickey Fuller test #after first diff
# tests for AR model with time trend
adfTest(dx, lags=3, type=c("ct"))
adfTest(dx, lags=7, type=c("ct"))

# tests for AR model with no time trend
adfTest(coredata(dx), lags=3, type=c("c"))
adfTest(coredata(dx), lags=7, type=c("c"))


#Unit-root tests on first difference
diffvar=diff(ats)
plot(diffvar)


#Finding model
#1 Bic
auto.arima(ats, ic =c("bic"), trace=TRUE, stationary = F)

#2 AIC
auto.arima(ats, ic =c("aic"), trace=TRUE, stationary = F)


#Searching sarima model
auto.arima(ats, trace=T, seasonal=T)


#Fitting ARMA model
m1=Arima(ats, order=c(1,1,1), method='ML', include.drift=T)
coeftest(m1)

# RESIDUAL ANALYSIS
plot(m1$residuals)
acf(coredata(m1$residuals), main="ACF plot of Residuals")

Box.test(m1$residuals,lag=3,type='Ljung-Box', fitdf=2)
Box.test(m1$residuals,lag=7,type='Ljung-Box', fitdf=2)

#histogram
hist(m1$residuals, xlab="Distribution", prob=TRUE, main="Histogram")
xfit<-seq(min(m1$residuals),max(m1$residuals),length=40)
yfit<-dnorm(xfit,mean=mean(m1$residuals),sd=sd(m1$residuals))
lines(xfit, yfit, col="blue", lwd=2)

# CREATE NORMAL PROBABILITY PLOT
qqnorm(m1$residuals)
qqline(m1$residuals, col = 2)

#FORECASTING
f=forecast(m1, h=5)
forecast(m1, h=5)

plot(f, include=50)

# BACKTESTING
source("backtest.R")
backtest(m1, ats, h=1, orig=length(price)*0.8)


######GARCH########################

#simple return time series
rets =(ats-lag(ats,k=-1))/lag(ats,k=-1)
plot(rets)
ret=coredata(rets)


#compute statistics
basicStats(rets)

#histogram
hist(rets, xlab="Distribution", prob=TRUE, main="Histogram")
xfit<-seq(min(rets),max(rets),length=40)
yfit<-dnorm(xfit,mean=mean(rets),sd=sd(rets))
lines(xfit, yfit, col="blue", lwd=2)

# Perform Jarque-Bera normality test. 
normalTest(rets,method=c("jb"))

par(mfrow=c(1,1))
# Plots ACF function of vector data
acf(ret)
# Plot ACF of squared returns to check for ARCH effect 
acf(ret^2)
# Plot ACF of absolute returns to check for ARCH effect 
acf(abs(ret))


#plot returns, square returns and abs(returns)
# Plots vector data
plot(rets, type='l')
# Plot squared returns to check for ARCH effect 
plot(rets^2,type='l')
# Plot absolute returns to check for ARCH effect 
plot(abs(rets),type='l')

# Computes Ljung-Box test on returns to test  independence 
Box.test(coredata(rets),lag=2,type='Ljung')
Box.test(coredata(rets),lag=4,type='Ljung')
Box.test(coredata(rets),lag=6,type='Ljung')
# Computes Ljung-Box test on squared returns to test non-linear independence 
Box.test(coredata(rets^2),lag=2,type='Ljung')
Box.test(coredata(rets^2),lag=4,type='Ljung')
Box.test(coredata(rets^2),lag=6,type='Ljung')
# Computes Ljung-Box test on absolute returns to test non-linear independence 
Box.test(abs(coredata(rets)),lag=2,type='Ljung')
Box.test(abs(coredata(rets)),lag=4,type='Ljung')
Box.test(abs(coredata(rets)),lag=6,type='Ljung')

#checking for ARMA order
auto.arima(rets, max.p = 2, max.q = 2, stationary=TRUE, ic=c("aic"), stepwise=TRUE)

###### Model 1#########
#Fit ARMA(0,0)-GARCH(1,1) model-normal distribution
m1.spec=ugarchspec(variance.model = list(garchOrder = c(1,1)), mean.model = list(armaOrder = c(0,0)))
m1.fit=ugarchfit(spec=m1.spec, data=rets)
m1.fit
#estimated coefficients:
coef(m1.fit)
#create selection list of plots for garch(1,1) fit
plot(m1.fit)
#conditional volatility plot
plot.ts(sigma(m1.fit), ylab="sigma(t)", col="blue")
#Compute information criteria using infocriteria() function for model selecton
infocriteria(m1.fit)



###### Model 2#########
#Fit ARMA(0,0)-GARCH(1,1) model with t-distribution
m1.t.spec=ugarchspec(variance.model=list(garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)), distribution.model = "std")
#estimate model 
m1.t.fit=ugarchfit(spec=m1.t.spec, data=rets)
m1.t.fit
#plot of residuals
plot(m1.t.fit)
infocriteria(m1.t.fit)


###### Model 3#########
#Fit ARMA(0,0)-eGARCH(1,1) model with normal-distribution
em1.spec=ugarchspec(variance.model=list(model = "eGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)))
#estimate model 
em1.fit=ugarchfit(spec=em1.spec, data=rets)
em1.fit
plot(em1.fit)


###### Model 4#########
#Fit ARMA(0,0)-eGARCH(1,1) model with t-distribution
em1.t.spec=ugarchspec(variance.model=list(model = "eGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)), distribution.model = "std")
#estimate model 
em1.t.fit=ugarchfit(spec=em1.t.spec, data=rets)
em1.t.fit
plot(em1.t.fit)


###### Model 5#########
#Fit ARMA(0,1)-eGARCH(1,1) model with t-distribution
em2.t.spec=ugarchspec(variance.model=list(model = "eGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,1)), distribution.model = "std")
#estimate model 
em2.t.fit=ugarchfit(spec=em2.t.spec, data=rets)
em2.t.fit
plot(em2.t.fit)



###### Model 6#########
#Fit ARMA(1,0)-eGARCH(1,1) model with t-distribution
em3.t.spec=ugarchspec(variance.model=list(model = "eGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(1,0)), distribution.model = "std")
#estimate model 
em3.t.fit=ugarchfit(spec=em3.t.spec, data=rets)
em3.t.fit
plot(em3.t.fit)
#get unconditional mean and variance
uncmean(em3.t.fit)
uncvariance(em3.t.fit)

# MODEL COMPARISON # compare information criteria 
model.list = list(m1 = m1.fit, m1.t = m1.t.fit, em1=em1.fit, em1.t = em1.t.fit, em2.t = em2.t.fit, em3.t=em3.t.fit)                      
info.mat = sapply(model.list, infocriteria) 
rownames(info.mat) = rownames(infocriteria(m1.fit)) 
info.mat 


# RE-FIT MODELS LEAVING 100 OUT-OF-SAMPLE OBSERVATIONS FOR FORECAST 
# EVALUATION STATISTICS 
m1.fit = ugarchfit(spec=m1.spec, data=ret, out.sample=100) 
m1.t.fit = ugarchfit(spec=m1.t.spec, data=ret, out.sample=100) 
em1.fit = ugarchfit(em1.spec, data=ret, out.sample=100) 
em1.t.fit = ugarchfit(em1.t.spec, data=ret, out.sample=100) 
em2.t.fit = ugarchfit(em2.t.spec, data=ret, out.sample=100) 
em3.t.fit = ugarchfit(em3.t.spec, data=ret, out.sample=100) 

# COMPUTE 100 1-STEP AHEAD ROLLING FORECASTS W/O RE-ESTIMATING 
m1.fcst = ugarchforecast(m1.fit, n.roll=100, n.ahead=1) 
m1.t.fcst = ugarchforecast(m1.t.fit, n.roll=100, n.ahead=1) 
em1.fcst = ugarchforecast(em1.fit, n.roll=100, n.ahead=1) 
em1.t.fcst = ugarchforecast(em1.t.fit, n.roll=100, n.ahead=1) 
em2.t.fcst = ugarchforecast(em2.t.fit, n.roll=100, n.ahead=1)
em3.t.fcst = ugarchforecast(em3.t.fit, n.roll=100, n.ahead=1) 


# COMPUTE FORECAST EVALUATION STATISTICS USING FPM() FUNCTION 

fcst.list = list(m1 = m1.fcst, m1.t = m1.t.fcst,em1=em1.fcst, em1.t = em1.t.fcst, em2.t=em2.t.fcst,em3.t=em3.t.fcst)  
fpm.mat = sapply(fcst.list, fpm) 
fpm.mat 


#FORECASTS for Model em3.t
#compute h-step ahead forecasts for h=1,2,...,10
em3.t.fit=ugarchfit(spec=em3.t.spec, data=rets, out.sample=100)
em3.fcst=ugarchforecast(em3.t.fit, n.ahead=10, n.roll=100)
em3.fcst
plot(em3.fcst)

