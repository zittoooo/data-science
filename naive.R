
install.packages("e1071") 
library(e1071)
?naiveBayes 
data(Titanic)
str(Titanic)

Titanic_df=as.data.frame(Titanic) 
str(Titanic_df) 

#Creating data from table 
#This will repeat each combination with the frequency of each combination 
sum(Titanic_df$Freq) 
repeating_sequence=rep.int(seq_len(nrow(Titanic_df)), Titanic_df$Freq) 

#Create the dataset by row repetition created 
Titanic_dataset=Titanic_df[repeating_sequence,] 

#We no longer need the frequency, drop the feature 
Titanic_dataset$Freq=NULL 

#Fitting the Naive Bayes model 
Naive_Bayes_Model=naiveBayes(Survived ~., data=Titanic_dataset) 

#What does the model say? Print the model summary 
Naive_Bayes_Model 

#Prediction on the dataset 
NB_Predictions=predict(Naive_Bayes_Model,Titanic_dataset) 

#Confusion matrix to check accuracy 
table(NB_Predictions,Titanic_dataset$Survived)


#Getting started with Naive Bayes in mlr
#Install the package 
install.packages("mlr")
library(mlr)

#Create a classification task for learning on Titanic Dataset 
#and specify the target feature 
task = makeClassifTask(data = Titanic_dataset, target = "Survived") 
task 


selected_model = makeLearner("classif.naiveBayes") 
#Train the model 
NB_mlr = train(selected_model, task) 
#Read the model learned 
NB_mlr$learner.model

#Predict on the dataset without passing the target feature 
predictions_mlr = as.data.frame(predict(NB_mlr, newdata = Titanic_dataset[,1:3])) 
#Confusion matrix to check accuracy 
table(predictions_mlr[,1],Titanic_dataset$Survived) 

#page 10

set.seed(1001) 
N <- 100 
x <- rnorm(N, mean = 3, sd = 2)
mean(x)
sd(x)
#> LL <- function(mu, sigma) { +     R = dnorm(x, mu, sigma) +     # +     -sum(log(R)) + }


library(stats4) 
mle(LL,start = list(mu = 1, sigma=1))  

#page 11
dnorm(x, 1, -1)

mle(LL, start = list(mu = 1, sigma=1), method = "L-BFGS-B", lower = c(Inf, 0), upper = c(Inf, Inf)) 

#> LL <- function(mu, sigma) { +     R = suppressWarnings(dnorm(x, mu, sigma)) +     # +     -sum(log(R)) + }
mle(LL, start = list(mu = 1, sigma=1)) 


#page 12
mle(LL, start = list(mu = 0, sigma=1)) 


#(2) Fitting a Linear Model 
x <- runif(N) 
y <- 5 * x + 3 + rnorm(N) 



fit <- lm(y ~ x) 
summary(fit) 

#page 13
plot(x, y) 
abline(fit, col = "red")
# LL <- function(beta0, beta1, mu, sigma) { +   # Find residuals +   # +   R = y - x * beta1 - beta0 +   # +   # Calculate the likelihood for the residuals (with mu and sigma as parameters) +   # +   R = suppressWarnings(dnorm(R, mu, sigma)) +   # +   # Sum the log likelihoods for all of the data points +   # +   -sum(log(R)) + } 
# LL <- function(beta0, beta1, mu, sigma) { +   R = y - x * beta1 - beta0 +   # +   R = suppressWarnings(dnorm(R, mu, sigma, log = TRUE)) +   # +   -sum(R) + }


#page 14
fit <- mle(LL, start = list(beta0 = 3, beta1 = 1, mu = 0, sigma=1)) 
fit <- mle(LL, start = list(beta0 = 5, beta1 = 3, mu = 0, sigma=1)) 
fit 

fit <- mle(LL, start = list(beta0 = 4, beta1 = 2, mu = 0, sigma=1))
fit 
summary(fit) 

fit <- mle(LL, start = list(beta0 = 2, beta1 = 1.5, sigma=1), fixed = list(mu = 0), nobs = length(y)) 
summary(fit) 
AIC(fit) 
BIC(fit) 
logLik(fit) 

#page 16

#library(bbmle)
#fit <- mle2(LL, start = list(beta0 = 3, beta1 = 1, mu = 0, sigma = 1)) 
#summary(fit) 








set.seed(123)
trueMean <- 10 
n <- 20
x <- rnorm(n, mean = trueMean) 
print(x) 
hist(x, col = "lavender") 
abline(v = mean(x), col = "red", lwd = 2) 





