import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns

dataset= pd.read_csv("RottenTomato/data/cleanRottenTomatoesMovies.csv")

feature_columns=["content_rating_as_int","tomatometer_rating","tomatometer_count","audience_rating","audience_count","tomatometer_top_critics_count","tomatometer_fresh_critics_count","tomatometer_rotten_critics_count"]
X=dataset[feature_columns]

y=dataset.tomatometer_status_results

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X, y)
y_pred = dummy_clf.predict(X)
print("Rotten Tomatoes Baseline Accuracy:", dummy_clf.score(X,y))


strategies = ['most_frequent', 'stratified', 'uniform', 'constant']
  
test_scores = []
for s in strategies:
    if s =='constant':
        dclf = DummyClassifier(strategy = s, random_state = 0, constant = 1)
    else:
        dclf = DummyClassifier(strategy = s, random_state = 0)
    dclf.fit(X_train, y_train)
    score = dclf.score(X_test, y_test)
    test_scores.append(score)


ax = sns.stripplot(strategies, test_scores);
ax.set(xlabel ='Strategy', ylabel ='Accuracy', title='Rotten Tomatoes Baseline Accuracy')
plt.show()