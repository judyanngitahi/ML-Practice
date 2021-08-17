# Gaussian naive bayes is a naive bayes algorithm that follows a gaussian normal dostribution and supports continuous data
# Assumes thatthe continuos value associated with each value are distributed in a normal distribution

#trying to use GNB to estimate if a person will buy a car or not. The information we have about them is their gender age and estimated salary

import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt     


#read in data to a df and check what it looks like
suv_data = pd.read_csv('/Users/judygitahi/Code/upgraded-potato/2-Gaussian_NB/suv_data.csv')
print(suv_data.head())

#the dataset we have currently has all values filled in so I split it into a test set and a training set
#x, the explanatory variables are age and estimated salary , and the outcome variable is the purchased column 

x = suv_data.iloc[:, [2, 3]].values 
y = suv_data.iloc[:, 4].values  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 5)

#any algorthim that uses gradient descent to optimize will require feature scaling to standardize between features
#in this case we need it scale age and estimated salary bith in the train and test datasets

scaler = StandardScaler()  
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 

classifier = GaussianNB()
classifier.fit(x_train, y_train)  

#once we have trained the model, I use that to predict new y values based on our x test values
y_pred = classifier.predict(x_test)  

#check the accuracy of the model
score = accuracy_score(y_test, y_pred) 
print(score)

# then we plot a confuison matrix, which has four tyoes of outcomes
# TP: True Positive: Predicted values correctly predicted as actual positive
# FP: False Positive: Predicted values incorrectly predicted an actual positive. i.e., Negative values predicted as positive
# FN: False Negative: Positive values predicted as negative
# TN: True Negative: Predicted values correctly predicted as an actual negative

matrix = confusion_matrix(
        y_test,
        y_pred
    )
print(matrix)


matrix2= plt.subplot()
sns.heatmap(matrix, annot=True, fmt='g', ax=matrix2)

# labels, title and ticks
matrix2.set_xlabel('Predicted labels')
matrix2.set_ylabel('True labels')
matrix2.set_title('Confusion Matrix')
matrix2.xaxis.set_ticklabels(['Not Purchased', 'Purchased']); matrix2.yaxis.set_ticklabels(['Not Purchased', 'Purchased'])
plt.show()

# results
# TP - 27 - meaning there were 27 people who purchased a car, and for whom we correctly predicted so 
# FP - 4 - we predicted that 4 people would buy cars yet they didint
# FN - 7 - we predicted that 7 people would not buy cars but they actually did buy them
# TN - 62 - we correctly predicted that 62 people would not buy cars