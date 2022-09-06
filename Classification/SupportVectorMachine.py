## Brief on Support Vector Machine
# Done it on the Rmd script

# Load the libraries 
import pandas as pd
import numpy as np

# Import the data
svc_data = pd.read_csv("../data/student_performance.csv")

# Data inspection
svc_data.info()

# Transformation
## Encode the categorical variables to numeric

svc_data = svc_data.replace({'address':{'R':0,'U':1},
                            'famsize':{'LE3':0,'GT3':1},
                            'Pstatus':{'P':0,'A':1},
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

svc_data.head()                            

