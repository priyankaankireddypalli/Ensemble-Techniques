# 3
library(readxl)
# Importing the dataset
cocoa <- read_xlsx("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Coca_Rating_Ensemble.xlsx")
# Performing EDA
attach(cocoa)
cocoa <- cocoa[c(-8,-9)]
str(cocoa)
summary(cocoa)

# Voting method
set.seed(12345)
Train_Test <- sample(c("Train", "Test"), nrow(cocoa), replace = TRUE, prob = c(0.7, 0.3))
voting_Train <- cocoa[Train_Test == "Train",]
voting_TestX <- within(cocoa[Train_Test == "Test", ], rm(Rating))
voting_TestY <- cocoa[Train_Test == "Test", "Rating"]
# Random Forest Analysis
library(randomForest)
voting_RF <- randomForest(Rating ~ ., data = voting_Train, keep.inbag = TRUE, ntree = 500)
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
cocoa <- read_xlsx("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Coca_Rating_Ensemble.xlsx")
# Performing EDA
cocoa <- cocoa[c(-8,-9)]
str(cocoa)
summary(cocoa)
cocoa$Company <- as.factor(cocoa$Company)
cocoa$Name <- as.factor(cocoa$Name)
cocoa$Company_Location <- as.factor(cocoa$Company_Location)
cocoa$Company <- as.integer(cocoa$Company)
cocoa$Name <- as.integer(cocoa$Name)
cocoa$Company_Location <- as.integer(cocoa$Company_Location)
library(caTools)
set.seed(0)
split <- sample.split(cocoa$Rating, SplitRatio = 0.8)
bagging_train <- subset(cocoa, split == TRUE)
bagging_test <- subset(cocoa, split == FALSE)
library(randomForest)
bagging <- randomForest(bagging_train$Rating ~ ., data = bagging_train, mtry=1:6)
# bagging will take all the columns ---> mtry = all the attributes
test_pred <- predict(bagging, bagging_test)
rmse_bagging <- sqrt(mean(bagging_test$Rating - test_pred)^2)
rmse_bagging
# Prediction for trained data result
train_pred <- predict(bagging, bagging_train)
# RMSE on Train Data
train_rmse <- sqrt(mean(bagging_train$Rating - train_pred)^2)
train_rmse

# Gradient boosting method
cocoa <- read_xlsx("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Coca_Rating_Ensemble.xlsx")
cocoa <- cocoa[c(-8,-9)]
# performing EDA
str(cocoa)
summary(cocoa)
cocoa$Company <- as.factor(cocoa$Company)
cocoa$Name <- as.factor(cocoa$Name)
cocoa$Company_Location <- as.factor(cocoa$Company_Location)
cocoa$Company <- as.integer(cocoa$Company)
cocoa$Name <- as.integer(cocoa$Name)
cocoa$Company_Location <- as.integer(cocoa$Company_Location)
library(caTools)
set.seed(0)
split <- sample.split(cocoa$Rating, SplitRatio = 0.8)
boosting_train <- subset(cocoa, split == TRUE)
boosting_test <- subset(cocoa, split == FALSE)
library(gbm)
boosting <- gbm(boosting_train$Rating ~ ., data = boosting_train, distribution = 'gaussian',n.trees = 5000, interaction.depth = 4, shrinkage = 0.2, verbose = F)
# distribution = Gaussian for regression and Bernoulli for classification
# Prediction for test data result
boost_test <- predict(boosting, boosting_test, n.trees = 5000)
rmse_boosting <- sqrt(mean(boosting_test$Rating - boost_test)^2)
rmse_boosting
# Prediction for train data result
boost_train <- predict(boosting, boosting_train, n.trees = 5000)
# RMSE on Train Data
rmse_train <- sqrt(mean(boosting_train$Rating - boost_train)^2)
rmse_train

# XG boosting method
cocoa <- read_xlsx("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Coca_Rating_Ensemble.xlsx")
cocoa <- cocoa[c(-8,-9)]
str(cocoa)
summary(cocoa)
cocoa$Company <- as.factor(cocoa$Company)
cocoa$Name <- as.factor(cocoa$Name)
cocoa$Company_Location <- as.factor(cocoa$Company_Location)
cocoa$Company <- as.integer(cocoa$Company)
cocoa$Name <- as.integer(cocoa$Name)
cocoa$Company_Location <- as.integer(cocoa$Company_Location)
library(caTools)
set.seed(0)
split <- sample.split(cocoa$Rating, SplitRatio = 0.8)
boosting_train <- subset(cocoa, split == TRUE)
boosting_test <- subset(cocoa, split == FALSE)
summary(boosting_train)
attach(boosting_train)
library(xgboost)
train_y <- boosting_train$Rating == "1"
str(boosting_train)
# creating dummy variables on attributes
train_x <- model.matrix(boosting_train$Rating ~ . -1, data = boosting_train)
test_y <- boosting_test$Rating == "1"
# Creating dummy variables on attributes
test_x <- model.matrix(boosting_test$Rating ~ . -1, data = boosting_test)
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
