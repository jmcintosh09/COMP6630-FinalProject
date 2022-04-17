import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.dummy import DummyClassifier

datasetX= pd.read_csv("data/X_data.csv")
datasetY= pd.read_csv("data/y_data.csv")

feature_columns=["runtime","title","studio","year","genre"]
X=datasetX[feature_columns]

y=datasetY.lifetime_gross

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train,y_train)
y_pred = dummy_clf.predict(X_test)

print("Box Office Revenue Baseline Accuracy:", dummy_clf.score(X,y))
print('Baseline Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))