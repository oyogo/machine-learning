## Brief on Support Vector Machine 
# Done it on the Regression Rmd script

# Load the libraries 
from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer


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

# Seeing that we want to model a classification model, we'll categorize the marks into two groups, 
# <10 will be a fail and >10 a pass. 
# Conditional replace
# We can make use of numpy function np.where 

svc_data['G3'] = np.where(svc_data['G3']>=10,1,0)

# Some columns are not really needed as such we'll drop them. 

svc_data.drop(['G1','reason','school'], axis=1,inplace=True)

# Data partitioning 
## response variable
y = svc_data['G3']
## predictor variables
X = svc_data.drop('G3',axis=1)

# Split the data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.2) 

# separate categorical columns from numeric ones. Further separate categorical columns into those 
# needing to be oneHot encoded and those that need to be ordinalEncoded 

num_cols = x_train.select_dtypes(exclude=['object']).columns.tolist() # extract numeric columns
cat_cols = x_train.select_dtypes(include=['object']).columns.tolist() # extract categorical columns 
ord_cols = ['Medu','Fedu','traveltime','studytime','famrel','freetime','goout','Dalc','Walc','health'] # categorical columns with order

# drop ordinal columns from the numeric columns list
num_cols = [value for value in num_cols if value not in ord_cols]
num_cols = [value for value in num_cols if value != "Unnamed: 0"]

# Make pipeline for numerical columns 
num_pipe = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

cat_pipe = make_pipeline(
    SimpleImputer(strategy='constant',fill_value='N/A'),
    OneHotEncoder(handle_unknown='ignore',sparse=False)
)

ord_pipe = make_pipeline(
    OrdinalEncoder()
)

# combine the pipelines
full_pipeline = ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('cat',cat_pipe,cat_cols),
    ('ord',ord_pipe,ord_cols)
])

# Make full pipeline 
svc_model = make_pipeline(full_pipeline, SVC(kernel='linear'))

## Train the model 
#svcclassifier = SVC(kernel='linear')

svc_model.fit(x_train,y_train)

 # Make predictions on the test data                         
y_pred = svc_model.predict(x_test)

# Use the accuracy_score method from sklearn metric module to calculate the accuracy.
print("Accuracy: ", accuracy_score(y_test,y_pred))

