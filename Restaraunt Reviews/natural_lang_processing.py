# importing the libraries

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# importing the dataset

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)
# x = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, -1].values

# Cleaning the dataset

nltk.download('stopwords')

corpus = []  # contains all different rewiews from dataset

for i in range(0, 1000):
    review = re.sub("[^a-zA-Z]", " ", dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word)
              for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)


# Creating the bag of words model

cv = CountVectorizer(max_features=1500)  # most frequent words
X = cv.fit_transform(corpus).toarray()  # gives a 2d array
y = dataset.iloc[:, -1].values
print(len(X[0]))

# Training and test set

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Training the naive bayes model  -- 67% accuracy

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Training the random forest classifier -- 71%accuracy

# classifier = RandomForestClassifier(n_estimators=100, criterion='entropy')
# classifier.fit(X_train, y_train)

# training the decision tree model  --72% accuracy

# classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
# classifier.fit(X_train, y_train)

# Training the SVM model  --75% accuracy

# classifier = SVC(kernel='rbf', random_state=0)
# classifier.fit(X_train, y_train)

# Training the knn model  --66% accuracy

# classifier = KNeighborsClassifier( n_neighbors=5, metric="minkowski", p=2)
# classifier.fit(X_train, y_train)

# Training the logistic regression model  --76% accuracy
# classifier = LogisticRegression(random_state=0)
# classifier.fit(X_train, y_train)


# predicting the test set results

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# confusion matrix and accuracy score

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# predicting the results of a single review

new_review = 'I love this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word)
              for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)
