### Importing Libraries
import pandas as pd

#Importing pandas-profiling package
from pandas_profiling import ProfileReport

import numpy as np
import warnings
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns




#Reading csv file (Data set file)
df = pd.read_csv('housing.csv')

train, test = train_test_split(df, train_size=0.9 ,test_size = 0.1, random_state=50, stratify=df['INDUS'])

classifier = RandomForestClassifier(n_estimators=100)
# Training the model
'''classifier.fit(X_train,y_train)'''
# predicting probabilities
rf_probs = classifier.predict_proba(X_test)
# keeping probabilities for the positive outcome only
rf_probs = rf_probs[:, 1]
# predicting 
y_pred=classifier.predict(X_test)




''''#printing csv file (Data set file)
print(df)

#Analaysis in html file
profile = ProfileReport(df)
profile.to_file(output_file="housing.html")'''

