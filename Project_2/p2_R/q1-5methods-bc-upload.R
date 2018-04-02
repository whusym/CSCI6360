# Project 2 for CSCI 6360, Spring 2018, University of Georgia
# Christian McDaniel and Jeremy Shi
# 5 different classication methods on the Wisconsin Breast Cancer dataset
# Targets: 2 for benign; 4 for malignant

library(foreign)
library(caret)
library(MASS)
library(bnlearn)
library(bnclassify)
library(class)
library(naivebayes)
library(e1071)
library(dplyr)
library(graph)

dataset <- read.csv("YOUR DATAPATH")
dataset <- dataset[, !(colnames(dataset) %in% c('X1000025'))]

# using the boilerplte code provided by Bahaa
score <- function(expected, predicted){
  tp = length(expected[which(expected == '4' & predicted == '4')]) + 1   # adding 1 to avoid NaN
  fp = length(expected[which(expected == '2' & predicted == '2')]) + 1    # adding 1 to avoid NaN
  fn = length(expected[which(expected == '4' & predicted == '4')]) + 1  # adding 1 to avoid NaN
  tn = length(expected[which(expected == '2' & predicted == '2')]) + 1    # adding 1 to avoid NaN
  
  accuracy = (tp + tn) / (length(expected) + 1)
  recall = tp / (tp + fn + 1)
  precision = tp / (tp + fp + 1)
  f.score = (2 * tp) / (2 * tp + fn + fp)
  
  r <- cbind(accuracy, recall, precision, f.score) 
  colnames(r)<- c('accuracy','recall','precision','fscore') 
  return(r)
}

trainAndTestFold <- function(fold, dataset) { 
  
  # Model 1: Naive Bayes
  dataset[] <- lapply(dataset, as.factor)
  training = dataset[-fold,]
  testing = dataset[fold,]
  model1 = naiveBayes(training$X2.1 ~., data=training)
  model1.pred = predict(model1, testing)
  model1.scores = score(testing$X2.1, model1.pred)
  
  # Model 2: TAN Bayes
  model2 = bnc('tan_cl', 'X2.1', training, smooth = 1)
  model2.pred = predict(model2, testing)
  model2.scores = score(testing$X2.1, model2.pred)
  
  # Model 3: LDA (Thanks Sean for reminding me the dataframe setting)
  dataset <- read.csv("YOUR DATA PATH")
  dataset <- dataset[, !(colnames(dataset) %in% c('X1000025'))]
  dataset <- as.data.frame(dataset)
  training = dataset[-fold,]
  testing = dataset[fold,]
  model3 = lda(formula = training$X2.1 ~., data=training)
  model3.pred = predict(model3, testing)
  model3.scores = score(testing$X2.1, model3.pred$class)
  
  # Model 4: Logistic Regression
  # From https://www.r-bloggers.com/how-to-perform-a-logistic-regression-in-r/
  dataset$X2.1 = ifelse(dataset$X2.1==4, 1, 0)
  training = dataset[-fold,]
  testing = dataset[fold,]
  model4 <- glm(formula=training$X2.1 ~., family=binomial(link='logit'), data=training)
  model4.pred = predict(model4, testing, type = "response")
  model4.pred = ifelse(model4.pred<0.5, '2', '4')
  testing$X2.1 = ifelse(testing$X2.1==0, '2', '4')
  model4.scores = score(testing$X2.1, model4.pred)
  
  # Model 5: KNN (K = 3)
  training = dataset[-fold,]
  testing = dataset[fold,]
  model5.pred <- knn(train = training, test = testing, cl = training$X2.1, k = 3)
  model5.pred = ifelse(model5.pred == 0, '2', '4')
  testing$X2.1 = ifelse(testing$X2.1==0, '2', '4')
  model5.scores = score(testing$X2.1, model5.pred)
  r <- rbind(model1.scores, model2.scores, model3.scores, model4.scores, model5.scores) 
  rownames(r)<- c('NaiveBayes', 'TANBayes', 'LDA', 'LR', 'KNN') 
  return(r)
}



nFolds <- 10
folds <- createFolds(dataset$X2.1, k = nFolds) 
cv<-lapply(folds, trainAndTestFold, dataset=dataset)
meanCV <- Reduce('+',cv) / nFolds
stdCV <- sqrt(Reduce('+',lapply(cv,function(x) (x - meanCV)^2)))
