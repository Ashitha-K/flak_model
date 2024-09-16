import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv('E:\\assignment_flask\\winequality-red (1).csv')


#treating the outliers
outlier_checking = ['total sulfur dioxide','free sulfur dioxide']
Q1 = data[outlier_checking].quantile(0.25)
Q3 = data[outlier_checking].quantile(0.75)
IQR = Q3 - Q1
outliers_lower = data[outlier_checking]<(Q1-1.5*IQR)
outliers_upper = data[outlier_checking]>(Q3 + 1.5*IQR)
#clip the outlier
data[outlier_checking] = data[outlier_checking].clip(lower = Q1-1.5*IQR, upper = Q3+1.5*IQR, axis = 1)
# Separate features and target variable
X = data.drop('quality', axis=1)
y = data['quality']
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale the data (important for SVM)
scaler = StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)
# ---- RANDOM FOREST ---- #
# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
# Make predictions with Random Forest
rf_y_pred = rf_model.predict(X_test)
# Initialize and train the Support Vector Machine model
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)
# Make predictions with SVM
svm_y_pred = svm_model.predict(X_test_scaled)
import pickle

with open('model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)