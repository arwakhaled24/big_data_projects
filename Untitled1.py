#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# URL of the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column names for the dataset
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Load the dataset into a DataFrame
data = pd.read_csv(url, names=column_names)

# Save the dataset to a CSV file
data.to_csv('heart_disease.csv', index=False)

print("Dataset downloaded and saved as heart_disease.csv")


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


# In[5]:


column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
data = pd.read_csv(url, header=None, names=column_names)
data.head()


# In[6]:


data.info()


# In[23]:


# Replace '?' with NaN
data.replace('?', np.nan, inplace=True)
# Convert all columns to numeric
data = data.apply(pd.to_numeric)
# Drop rows with missing values
data.dropna(inplace=True)
print(data)


# In[24]:


# Plotting Histogram with Seaborn
plt.figure(figsize=(8, 6))
sns.histplot(data['age'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[25]:


# Line Chart for 'thalach' (Maximum Heart Rate Achieved)
plt.figure(figsize=(8, 6))
plt.plot(data['thalach'], color='green')
plt.title('Maximum Heart Rate Achieved (thalach)')
plt.xlabel('Index')
plt.ylabel('thalach')
plt.show()


# In[26]:


sex_target_counts = data.groupby(['sex', 'target']).size().unstack()
sex_target_counts.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('Distribution of Heart Disease by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Target', loc='upper right')
plt.show()


# In[8]:


features = data.drop(columns=['target'])
target = data['target'].apply(lambda x: 1 if x > 0 else 0)  # Binary classification


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# In[10]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(data)


# In[11]:


models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Neural Network": MLPClassifier(max_iter=500)
}
print(models)


# In[15]:


def evaluate_model(actual, predicted):
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    return precision, recall, accuracy


# In[16]:


svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)


# In[17]:


knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)


# In[18]:


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


# In[19]:


def calculate_metrics(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    return precision, recall, accuracy

precision_svm, recall_svm, accuracy_svm = calculate_metrics(y_test, y_pred_svm)
precision_knn, recall_knn, accuracy_knn = calculate_metrics(y_test, y_pred_knn)
precision_rf, recall_rf, accuracy_rf = calculate_metrics(y_test, y_pred_rf)


# In[20]:


# Print metrics
print(f"SVM - Precision: {precision_svm:.4f}, Recall: {recall_svm:.4f}, Accuracy: {accuracy_svm:.4f}")
print(f"KNN - Precision: {precision_knn:.4f}, Recall: {recall_knn:.4f}, Accuracy: {accuracy_knn:.4f}")
print(f"Random Forest - Precision: {precision_rf:.4f}, Recall: {recall_rf:.4f}, Accuracy: {accuracy_rf:.4f}")


# In[21]:


metrics = {
    'Model': ['SVM', 'KNN', 'Random Forest'],
    'Precision': [precision_svm, precision_knn, precision_rf],
    'Recall': [recall_svm, recall_knn, recall_rf],
    'Accuracy': [accuracy_svm, accuracy_knn, accuracy_rf]
}

metrics_df = pd.DataFrame(metrics)

plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='Precision', data=metrics_df)
plt.title('Model Precision')
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='Recall', data=metrics_df)
plt.title('Model Recall')
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='Accuracy', data=metrics_df)
plt.title('Model Accuracy')
plt.show()


# In[ ]:




