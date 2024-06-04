import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Separate features and target variable
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
LogReg = LogisticRegression(max_iter=1000)
LogReg.fit(X_train_scaled, y_train)

# Create and train the decision tree model
DecisionTree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
DecisionTree.fit(X_train, y_train)

# Create and train the Naive Bayes model
NaiveBayes = GaussianNB()
NaiveBayes.fit(X_train, y_train)

# Create and train the random forest model
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(X_train, y_train)

# Save the models using pickle
pickle.dump(LogReg, open('LogReg_model.pkl', 'wb'))
pickle.dump(DecisionTree, open('DecisionTree_model.pkl', 'wb'))
pickle.dump(NaiveBayes, open('NaiveBayes_model.pkl', 'wb'))
pickle.dump(RF, open('RF_model.pkl', 'wb'))
