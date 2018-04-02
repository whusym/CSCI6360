# Project 2 for CSCI 6360, Spring 2018, University of Georgia
# Christian McDaniel and Jeremy Shi
# 5 different classication methods on the Caravan dataset

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

# using the boilerplte code provided by Bahaa
score <- function(expected, predicted){
  tp = length(expected[which(expected == 'Yes' & predicted == 'Yes')]) + 1   # adding 1 to avoid NaN
  fp = length(expected[which(expected == 'No' & predicted == 'No')]) + 1    # adding 1 to avoid NaN
  fn = length(expected[which(expected == 'Yes' & predicted == 'Yes')]) + 1  # adding 1 to avoid NaN
  tn = length(expected[which(expected == 'No' & predicted == 'No')]) + 1    # adding 1 to avoid NaN
  
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
  model1 = naiveBayes(training$Purchase ~., data=training)
  model1.pred = predict(model1, testing)
  model1.scores = score(testing$Purchase, model1.pred)
  
  # Model 2: TAN Bayes
  model2 = bnc('tan_cl', 'Purchase', training, smooth = 1)
  model2.pred = predict(model2, testing)
  model2.scores = score(testing$Purchase, model2.pred)
  
  # Model 3: LDA (Thanks Sean for reminding me the dataframe setting)
  dataset <- read.csv("YOUR DATA PATH")
  dataset <- as.data.frame(dataset)
  training = dataset[-fold,]
  testing = dataset[fold,]
  model3 = lda(formula = training$Purchase ~., data=training)
  model3.pred = predict(model3, testing)
  model3.scores = score(testing$Purchase, model3.pred$class)
  
  # Model 4: Logistic Regression
  # From https://www.r-bloggers.com/how-to-perform-a-logistic-regression-in-r/
  dataset$Purchase = ifelse(dataset$Purchase=="Yes", 1, 0)
  training = dataset[-fold,]
  testing = dataset[fold,]
  model4 <- glm(formula=training$Purchase ~., family=binomial(link='logit'), data=training)
  model4.pred = predict(model4, testing, type = "response")
  model4.pred = ifelse(model4.pred<0.5, 'No', 'Yes')
  testing$Purchase = ifelse(testing$Purchase==0, 'No', 'Yes')
  model4.scores = score(testing$Purchase, model4.pred)
  
  # Model 5: KNN (K = 3)
  training = dataset[-fold,]
  testing = dataset[fold,]
  model5.pred <- knn(train = training, test = testing, cl = training$Purchase, k = 3)
  model5.pred = ifelse(model5.pred == 0, 'No', 'Yes')
  testing$Purchase = ifelse(testing$Purchase==0, 'No', 'Yes')
  model5.scores = score(testing$Purchase, model5.pred)
  r <- rbind(model1.scores, model2.scores, model3.scores, model4.scores, model5.scores) 
  rownames(r)<- c('NaiveBayes', 'TANBayes', 'LDA', 'LR', 'KNN') 
  return(r)
}

nFolds <- 10
folds <- createFolds(dataset$Purchase, k = nFolds) 
cv<-lapply(folds, trainAndTestFold, dataset=dataset)
meanCV <- Reduce('+',cv) / nFolds
stdCV <- sqrt(Reduce('+',lapply(cv,function(x) (x - meanCV)^2)))