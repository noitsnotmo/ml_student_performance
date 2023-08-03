
import tensorflow

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


data = pd.read_csv("student_mat.csv", sep=";") # open data csv file

# print(data.head()) # see data frame

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] # trim data


predict = "G3" # separate our data into separate arrays


x = np.array(data.drop([predict], axis=1)) # features

y = np.array(data[predict]) # labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)


linear = linear_model.LinearRegression() # define the model we will be using

linear.fit(x_train, y_train)

acc = linear.score(x_test, y_test) # acc stands for accuracy

print(acc) # score of above 80% fairly good for this data set

print('Coefficient: \n', linear.coef_) # slope values
print('Intercept \n', linear.intercept_) # intercept

predictions = linear.predict(x_test) # gets list of all predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

