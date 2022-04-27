import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

dataset= pd.read_csv("RottenTomato/data/cleanRottenTomatoesMovies.csv")

feature_columns=["content_rating_as_int","tomatometer_rating","tomatometer_count","audience_rating","audience_count","tomatometer_top_critics_count","tomatometer_fresh_critics_count","tomatometer_rotten_critics_count"]
X=dataset[feature_columns]

y=dataset.tomatometer_status_results

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X,y)
dummy_clf.predict(X)
print("Rotten Tomatoes Baseline Accuracy:", dummy_clf.score(X,y))