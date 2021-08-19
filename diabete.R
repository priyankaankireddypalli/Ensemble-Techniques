# 1
library(readr)
# Importing the dataset
Diabetes <- read.csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Diabeted_Ensemble.csv",stringsAsFactors = TRUE)
View(Diabetes)
# Performing EDA
attach(Diabetes)
str(Diabetes)
# Voting method
set.seed(12345)
Train_Test <- sample(c("Train", "Test"), nrow(Diabetes), replace = TRUE, prob = c(0.7, 0.3))
voting_Train <- Diabetes[Train_Test == "Train",]
voting_TestX <- within(Diabetes[Train_Test == "Test", ], rm(Class.variable))
voting_TestY <- Diabetes[Train_Test == "Test", "Class.variable"]
# Random Forest Analysis
library(randomForest)
voting_RF <- randomForest(Class.variable ~ ., data = voting_Train, keep.inbag = TRUE, ntree = 500)
# Overall class prediction (hard voting)
voting_RF_Test_Margin <- predict(voting_RF, newdata = voting_TestX, type = "class")
# Prediction
voting_RF_Test_Predict <- predict(voting_RF, newdata = voting_TestX, type = "class", predict.all = TRUE)
sum(voting_RF_Test_Margin == voting_RF_Test_Predict$aggregate)
head(voting_RF_Test_Margin == voting_RF_Test_Predict$aggregate)
# Majority Voting
dim(voting_RF_Test_Predict$individual)
Row_Count_Max <- function(x) names(which.max(table(x)))
Voting_Predict <- apply(voting_RF_Test_Predict$individual, 1, Row_Count_Max)
head(Voting_Predict)
tail(Voting_Predict)
all(Voting_Predict == voting_RF_Test_Predict$aggregate)
all(Voting_Predict == voting_RF_Test_Margin)
mean(Voting_Predict == voting_TestY)

# Bagging method
# Performing EDA
str(Diabetes)
Class.variable <- as.factor(Class.variable)
Class.variable <- as.integer(Class.variable)
library(caTools)
set.seed(0)
split <- sample.split(Diabetes$Class.variable, SplitRatio = 0.8)
bagging_train <- subset(Diabetes, split == TRUE)
bagging_test <- subset(Diabetes, split == FALSE)
library(randomForest)
bagging <- randomForest(bagging_train$Class.variable ~ ., data = bagging_train, mtry = 8)
# bagging will take all the columns ---> mtry = all the attributes
test_pred <- predict(bagging, bagging_test)
rmse_bagging <- sqrt(mean(bagging_test$Class.variable - test_pred)^2)
rmse_bagging
# Prediction for trained data result
train_pred <- predict(bagging, bagging_train)
# RMSE on Train Data
train_rmse <- sqrt(mean(bagging_train$Class.variable - train_pred)^2)
train_rmse

# Gradient boosting method
# Performing EDA
str(Diabetes)
Class.variable <- as.integer(Class.variable)
library(caTools)
set.seed(0)
split <- sample.split(Diabetes$Class.variable, SplitRatio = 0.8)
boosting_train <- subset(Diabetes, split == TRUE)
boosting_test <- subset(Diabetes, split == FALSE)
install.packages("gbm")
library(gbm)
boosting <- gbm(boosting_train$Class.variable ~ ., data = boosting_train, distribution = 'gaussian',n.trees = 5000, interaction.depth = 4, shrinkage = 0.2, verbose = F)
# distribution = Gaussian for regression and Bernoulli for classification
# Prediction for test data result
boost_test <- predict(boosting, boosting_test, n.trees = 5000)
rmse_boosting <- sqrt(mean(boosting_test$Class.variable - boost_test)^2)
rmse_boosting
# Prediction for train data result
boost_train <- predict(boosting, boosting_train, n.trees = 5000)
# RMSE on Train Data
rmse_train <- sqrt(mean(boosting_train$Class.variable - boost_train)^2)
rmse_train

# XG boosting method
# Performing EDA
str(Diabetes)
library(caTools)
set.seed(0)
split <- sample.split(Diabetes$Class.variable, SplitRatio = 0.8)
boosting_train <- subset(Diabetes, split == TRUE)
boosting_test <- subset(Diabetes, split == FALSE)
summary(boosting_train)
attach(boosting_train)
install.packages("xgboost")
library(xgboost)
train_y <- boosting_train$Class.variable == "1"
str(boosting_train)
# Creating dummy variables on attributes
train_x <- model.matrix(boosting_train$Class.variable ~ . -1, data = boosting_train)
test_y <- boosting_test$Class.variable == "1"
# Creating dummy variables on attributes
test_x <- model.matrix(boosting_test$Class.variable ~ . -1, data = boosting_test)
# DMatrix on train
Xmatrix_train <- xgb.DMatrix(data = train_x, label = train_y)
# DMatrix on test 
Xmatrix_test <- xgb.DMatrix(data = test_x, label = test_y)
# Max number of boosting iterations - nround
xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,objective = "multi:softmax", eta = 0.3, num_class = 2, max_depth = 100)
# Prediction for test data
xgbpred_test <- predict(xg_boosting, Xmatrix_test)
table(test_y, xgbpred_test)
mean(test_y == xgbpred_test)
# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)
table(train_y, xgbpred_train)
mean(train_y == xgbpred_train)
