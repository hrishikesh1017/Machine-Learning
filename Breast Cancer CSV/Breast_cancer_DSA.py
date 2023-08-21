# Importing Necessary Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier

# Data preprocessing


dataset=pd.read_csv("breast-cancer.csv")
dataset = dataset.drop(columns="id", axis=1)


x=dataset.iloc[:,1: ].values
y=dataset.iloc[:,0].values

print(x)

le=LabelEncoder()
y=le.fit_transform(y)


# Splitting the dataset into training and test sets

X_train,X_test,Y_train,Y_test=train_test_split(x, y, test_size=0.2,random_state=15)

# Feature Scaling

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

pca = PCA(n_components=3)
x_train = pca.fit_transform(X_train)
x_test = pca.transform(X_test)


# Training

classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(X_train, Y_train)

# predicting test set results

y_pred = classifier.predict(X_test)

# making the confusion matrix


print(confusion_matrix(Y_test, y_pred))
print(accuracy_score(Y_test, y_pred))

# Visualizing

from sklearn.metrics import roc_curve, roc_auc_score

# Get the predicted probabilities
y_probs = classifier.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(Y_test, y_probs)
roc_auc = roc_auc_score(Y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()