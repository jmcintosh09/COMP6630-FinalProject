import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler, LabelEncoder


metadata = pd.read_csv('box_office_revenue/data/movies_metadata.csv')
data = pd.read_csv('box_office_revenue/data/boxoffice.csv')

metadata = metadata.loc[:, metadata.columns.intersection(['genres', 'title', 'runtime'])]
df = metadata.merge(data, on='title')
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.reset_index()

y = df['lifetime_gross']
X = df.drop('lifetime_gross', axis=1)

genres_arr = []
for i in df['genres']:
    i = i.replace("'",'"')
    genre = json.loads(i)
    if genre and genre[0]:
        genre = genre[0]["name"]
    genres_arr.append(genre)

X['genre'] = genres_arr
X = X.drop(['genres','rank'], axis=1)
new_X = X.copy()

le = LabelEncoder()
X['title'] = le.fit_transform(X['title'])
X['genre'] = le.fit_transform(X['genre'].astype(str))
X['studio'] = le.fit_transform(X['studio'])
X['year'] = le.fit_transform(X['year'])

y = y.to_numpy().reshape(-1,1)

y_scaler = StandardScaler().fit(y)
y = y_scaler.transform(y)

x_scaler = StandardScaler().fit(X)
X = x_scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the Random Forest Model 
rf_model = RandomForestRegressor(random_state=1)

# Fitting the model 
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
# Calculating feature importance
importances = rf_model.feature_importances_
feature_names = [f"{col}" for col in new_X.columns[:]]
forest_importances = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

print('Random Forest Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Random Forest Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Random Forest Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Multi Layer Perceptron with 50000 iterations
regr = MLPRegressor(random_state=1, max_iter=50000).fit(X_train, y_train)
print(f"\nMulti Layer Perceptron Score:  {regr.score(X_test, y_test)}")

# Linear Regression
reg = LinearRegression().fit(X, y)
print(f"\nLinear Regression score: {reg.score(X, y)}")

print(y_train[0])

regr_svm = svm.SVR()
regr_svm.fit(X_train, y_train)
y_pred = regr_svm.predict(X_test)
print('\nSVM Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('SVM Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('SVM Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

predictions = regr_svm.predict(X)
predictions = predictions.reshape(-1,1)
predictions = y_scaler.inverse_transform(predictions)

new_X['box office predictions'] = predictions

new_X.to_csv('box_office_revenue/data/predictions_data.csv')
