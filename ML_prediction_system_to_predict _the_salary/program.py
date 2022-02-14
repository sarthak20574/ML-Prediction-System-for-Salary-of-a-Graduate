
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#loading the data
data_=pd.read_csv('./Data for IIS-Lab Assignment.csv')
#printing top 5 rows
print(data_.head(5))

# print no of columns, rows ,type of columns
#data_.info()

log_reg=LogisticRegression( max_iter=100000)

# with everything except gender, DOB, id, 10Board, 12Board,CollegeCityID,computerprograming,mechanicalengg...
print("\nWith everything: ")
x=data_[['10percentage','CollegeID','12graduation','12percentage','collegeGPA',
                'English','Quant','Domain','ElectronicsAndSemicon',
                'ComputerScience','ElectricalEngg','TelecomEngg','CivilEngg','conscientiousness',
                'agreeableness','extraversion','nueroticism','openess_to_experience','GraduationYear']]
y=data_['High-Salary']


#testingdata =20% train=80%
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
#print(len(x_train))
log_reg.fit(x_train,y_train)
# print("coefficient: "+str(log_reg.coef_))
# print("Intercept: "+str(log_reg.intercept_))
#log_reg.predict(x_test)
#print(y_test)
#accuracy
print("accuracyfor 80:20="+str(log_reg.score(x_test,y_test)))

print("\nconfusion matrix: ")
#storing predicted data
predicted=log_reg.predict(x_test)
c_matrix=confusion_matrix(y_test, predicted)
print(c_matrix)
print("\n nomalized confusion matrix: ")
print(c_matrix.diagonal()/c_matrix.sum(axis=1))

print("\n Classification Report:")
print(classification_report(y_test, predicted))

print("\nPrecision:",sklearn.metrics.precision_score(y_test, predicted))
print("Recall:",sklearn.metrics.recall_score(y_test, predicted))
