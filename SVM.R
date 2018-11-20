#(1) Linear SVM Classifier 

set.seed(10111) 
x = matrix(rnorm(40), 20, 2) 
y = rep(c(-1, 1), c(10, 10)) 
x[y == 1,] = x[y == 1,] + 1 
plot(x, col = y + 3, pch = 19) 

library(e1071)
dat = data.frame(x, y = as.factor(y)) 
svmfit = svm(y ~ ., data = dat, kernel = "linear", cost = 10, scale = FALSE)
print(svmfit)
plot(svmfit, dat)


#Non-Linear SVM Classifier

load(file = "ESL.mixture.rda") 
names(ESL.mixture) 
rm(x, y)
attach(ESL.mixture)
plot(x, col = y + 1)

dat = data.frame(y = factor(y), x)
fit = svm(factor(y) ~ ., data = dat, scale = FALSE, kernel = "radial", cost = 5) 
xgrid = expand.grid(X1 = px1, X2 = px2)
ygrid = predict(fit, xgrid)
plot(xgrid, col = as.numeric(ygrid), pch = 20, cex = .2) 
points(x, col = y + 1, pch = 19) 

func = predict(fit, xgrid, decision.values = TRUE)
func = attributes(func)$decision

xgrid = expand.grid(X1 = px1, X2 = px2) 
ygrid = predict(fit, xgrid) 
plot(xgrid, col = as.numeric(ygrid), pch = 20, cex = .2) 
points(x, col = y + 1, pch = 19) 
contour(px1, px2, matrix(func, 69, 99), level = 0, add = TRUE) 
contour(px1, px2, matrix(func,69,99), level = 0.5, add = TRUE, col = "blue", lwd = 2) 
