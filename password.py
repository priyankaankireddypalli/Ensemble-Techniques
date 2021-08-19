# 4
import pandas as pd
import numpy as np
# Importing the dataset
password = pd.read_excel("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Ensemble_Password_Strength.xlsx")
password.describe()
password.columns
password.info()
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
# Creating the Dummy variables
# n-1 dummy variables will be created for n categories
password = pd.get_dummies(password, columns = ['characters'], drop_first = True)
# Input and Output Split
predictors = password.loc[:, password.columns!='characters_strength']
target =  password['characters_strength']
# Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)
# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)
# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_1),('Prc', learner_2),('SVM', learner_3)])
# Fit classifier with the training data
voting.fit(x_train, y_train)
# Predict the most voted class
hard_predictions = voting.predict(x_test)
# Accuracy of hard voting
print('Hard Voting:', accuracy_score(y_test, hard_predictions))
# Soft Voting 
# Instantiate the learners (classifiers)
learner_4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
learner_5 = naive_bayes.GaussianNB()
learner_6 = svm.SVC(gamma = 0.001, probability = True)
# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_4),('NB', learner_5),('SVM', learner_6)],voting = 'soft')
# Fit classifier with the training data
voting.fit(x_train, y_train)
learner_4.fit(x_train, y_train)
learner_5.fit(x_train, y_train)
learner_6.fit(x_train, y_train)
# Predict the most probable class
soft_predictions = voting.predict(x_test)
# Get the base learner predictions
predictions_4 = learner_4.predict(x_test)
predictions_5 = learner_5.predict(x_test)
predictions_6 = learner_6.predict(x_test)
# Accuracies of base learners
print('L4:', accuracy_score(y_test, predictions_4))
print('L5:', accuracy_score(y_test, predictions_5))
print('L6:', accuracy_score(y_test, predictions_6))
# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(y_test, soft_predictions))

# Bagging method
import pandas as pd
import numpy as np
password = pd.read_excel("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Ensemble_Password_Strength.xlsx")
password.describe()
password.columns
password.info()
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
# Creating the Dummy variables
# n-1 dummy variables will be created for n categories
password = pd.get_dummies(password, columns = ['characters'], drop_first = True)
# Input and Output Split
predictors = password.loc[:, password.columns!='characters_strength']
target =  password['characters_strength']
# Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)
from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 30,bootstrap = True, n_jobs = 1, random_state = 42)
bag_clf.fit(x_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix
# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))
# Evaluation on Training Data
confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train))

# Gradient boosting method
import pandas as pd
import numpy as np
password = pd.read_excel("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Ensemble_Password_Strength.xlsx")
password.describe()
password.columns
password.info()
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
# Creating the Dummy variables
# n-1 dummy variables will be created for n categories
password = pd.get_dummies(password, columns = ['characters'], drop_first = True)
# Input and Output Split
predictors = password.loc[:, password.columns!='characters_strength']
target =  password['characters_strength']
# Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)
from sklearn.ensemble import GradientBoostingClassifier
boost_clf = GradientBoostingClassifier()
boost_clf.fit(x_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix
# Evaluation on Testing Data befor tuning hyperparameters
confusion_matrix(y_test, boost_clf.predict(x_test))
accuracy_score(y_test, boost_clf.predict(x_test))
# Evaluation on Training Data befor tuning hyperparameters
confusion_matrix(y_train, boost_clf.predict(x_train))
accuracy_score(y_train, boost_clf.predict(x_train))

# Hyperparameters tuning
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 100, max_depth = 1)
boost_clf2.fit(x_train, y_train)
# Evaluation on Testing Data after tuning hyperparameters
confusion_matrix(y_test, boost_clf2.predict(x_test))
accuracy_score(y_test, boost_clf2.predict(x_test))
# Evaluation on Training Data after tuning hyperparameters
confusion_matrix(y_train, boost_clf.predict(x_train))
accuracy_score(y_train, boost_clf.predict(x_train))
