## Brief on Support Vector Machine 
# Done it on the Regression Rmd script

# Load the libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

# Import the data
svc_data = pd.read_csv("../data/student_performance.csv")

# Data inspection
svc_data.info()

# Data transformation
## Encode the categorical variables to numeric

svc_data1 = svc_data.replace({'address':{'R':0,'U':1},
                            'famsize':{'LE3':0,'GT3':1},
                            'Pstatus':{'T':0,'A':1},
                            'Mjob':{'teacher':0,'at_home':1,'services':2,'other':3,'health':4},
                            'Fjob':{'teacher':0,'at_home':1,'services':2,'other':3,'health':4},
                            'guardian':{'mother':0,'father':1,'other':2},
                            'schoolsup':{'yes':1,'no':0},
                            'famsup':{'yes':1,'no':0},
                            'paid':{'no':0,'yes':1},
                            'activities':{'no':0,'yes':1},
                            'nursery':{'no':0,'yes':1},
                            'higher':{'yes':1,'no':0},
                            'internet':{'no':0,'yes':1},
                            'romantic':{'no':0,'yes':1},
                            'sex':{'F':0,'M':1}
                            }, regex=True)

svc_data2 = svc_data.replace({'address':{'R':0,'U':1},
                            'famsize':{'LE3':0,'GT3':1},
                            'Pstatus':{'T':0,'A':1},
                            'guardian':{'mother':0,'father':1,'other':2},
                            'schoolsup':{'yes':1,'no':0},
                            'famsup':{'yes':1,'no':0},
                            'paid':{'no':0,'yes':1},
                            'activities':{'no':0,'yes':1},
                            'nursery':{'no':0,'yes':1},
                            'higher':{'yes':1,'no':0},
                            'internet':{'no':0,'yes':1},
                            'romantic':{'no':0,'yes':1},
                            'sex':{'F':0,'M':1}
                            }, regex=True)
# Seeing that we want to model a classification model, we'll categorize the marks into two groups, 
# <10 will be a fail and >10 a pass. 
# Conditional replace
# We can make use of numpy function np.where 

svc_data2['G3'] = np.where(svc_data2['G3']>=10,1,0)

# Some columns are not really needed as such we'll drop them. 

svc_data2.drop(['G1','reason','school'], axis=1,inplace=True)

# Data partitioning 
## response variable
y = svc_data2['G3']
## predictor variables
X = svc_data2.drop('G3',axis=1)

# Onehot encoding - categorical variables 

one_hotFjob = pd.get_dummies(svc_data2.Fjob, prefix='Fjob')
one_hotMjob = pd.get_dummies(svc_data2.Mjob, prefix="Mjob")
print(one_hotFjob.head())

svc_data2 = svc_data2.drop('Fjob',axis=1)
svc_data2 = svc_data2.drop('Mjob',axis=1)

svc_data2 = svc_data2.join(one_hotFjob)
svc_data2 = svc_data2.join(one_hotMjob) 


y = svc_data2['G3']
x = svc_data2.drop('G3',axis=1)

svc_data2.head()

# Split the data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.2) 

## Fit the model 
svcclassifier = SVC(kernel='linear')

svcclassifier.fit(x_train,y_train)

 # Make predictions on the test data                         
y_pred = svcclassifier.predict(x_test)

# Use the accuracy_score method from sklearn metric module to calculate the accuracy.
print("Accuracy: ", accuracy_score(y_test,y_pred))

