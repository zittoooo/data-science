str(iris)
set.seed(1234)
ind <- sample(2,nrow(iris),replace=TRUE, prob=c(0.7, 0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]

library(party)
myFormual <-Species ~ Sepal.Length + Sepal.Width + Petal.Length +Petal.Width
iris_ctree <- ctree(myFormual, data = trainData)
table(predict(iris_ctree), trainData$Species)

print(iris_ctree)
plot(iris_ctree)
plot(iris_ctree, type = "simple")

#predict on test data
testPred <- predict(iris_ctree, newdata = testData)
table(testPred, testData$Species)

# 2)decision trees with package rpart
install.packages("TH.data")
data("bodyfat", package = "TH.data")
dim(bodyfat)

attributes(bodyfat)

bodyfat[1:5,]



set.seed(1234)
ind <- sample(2, nrow(bodyfat), replace = TRUE, prob= c(0.7, 0.3))
bodyfat.train <- bodyfat[ind==1,]
bodyfat.test <- bodyfat[ind==2,]
#train a decision tree
library(rpart)
myFormula <- DEXfat ~ age + waistcirc + hipcirc + elbowbreadth + kneebreadth
#bodyfat_rpart <- rpart(myFormula, data= bodyfat.train, 
#+                       control = rpart.control(minsplit = 10))
attributes(bodyfat_rpart)

print(bodyfat_rpart$cptable)
print(bodyfat_rpart)

plot(bodyfat_rpart)
text(bodyfat_rpart, use.n = T)

opt <- which.min(bodyfat_rpart$cptable[,"xerror"])
cp <- bodyfat_rpart$cptable[opt, "CP"]
bodyfat_prune <- prune(bodyfat_rpart, cp = cp)

print(bodyfat_prune)

plot(bodyfat_prune)
text(bodyfat_prune, use.n=T)

DEXfat_pred <- predict(bodyfat_prune, newdata = bodyfat.test)
xlim <- range(bodyfat$DEXfat)

abline(a= 0, b= 1)

#3)Random Forest
int <- sample(2, nrow(iris), replace = TRUE, prob = c(0.7,0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]

library(randomForest)
rf <- randomForest(Species ~ ., data = trainData, ntree=100, proximity=TRUE)
table(predict(rf), trainData$Species) 
print(rf)

attributes(rf)

plot(rf)

importance(rf)

varImpPlot(rf)

irisPred <- predict(rf, newdata=testData)
table(irisPred, testData$Species)

plot(margin(rf,testData$Species))


# 4)ROCR 패키지로 성과분석
install.packages("party")
install.packages("ROCR")

library(rpart)
x<-kyphosis[sample(1:nrow(kyphosis), nrow(kyphosis), replace = F),]
x.train <- kyphosis[1:floor(nrow(x)*.75),]
x.evaluate <- kyphosis[(floor(nrow(x)*.75)+1) : nrow(x),]
library(party)
xmodel <- cforest(Kyphosis ~ Age + Number + Start, data = x.train, 
                  control = cforest_unbiased(mtry = 3))
x.model <- ctree(Kyphosis ~ Age + Number + Start, data = x.train)
plot(x.model)
x.evaluate$prediction <- predict(x.model, newdata=x.evaluate)
x.evaluate$correct <- x.evaluate$prediction == x.evaluate$Kyphosis
print(paste("% of predicted classifications correct", mean(x.evaluate$correct)))

x.evaluate$probabilities <- 1- unlist(treeresponse(x.model, newdata=x.evaluate),
                                     use.names=F)[seq(1,nrow(x.evaluate)*2,2)]

library(ROCR)
pred <- prediction(x.evaluate$probabilities, x.evaluate$Kyphosis)
perf <- performance(pred, "tpr","fpr")
plot(perf, main = "ROC curve", colorize= T)
perf <- performance(pred,"lift","rpp")
plot(perf, main="lift curve", colorixze = T)

