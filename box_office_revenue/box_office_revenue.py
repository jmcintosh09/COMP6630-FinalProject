import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler


X = pd.read_csv("box_office_revenue/data/X_data.csv")
y = pd.read_csv('box_office_revenue/data/y_data.csv')
new_X = pd.read_csv('box_office_revenue/data/new_X_data.csv')

# Final Cleaning
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
feat_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feat_importances.nlargest().plot(kind='barh')

print('Random Forest Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Random Forest Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Random Forest Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Multi Layer Perceptron with 50000 iterations
regr = MLPRegressor(random_state=1, max_iter=50000).fit(X_train, y_train)
print(f"Multi Layer Perceptron Score:  {regr.score(X_test, y_test)}")

reg = LinearRegression().fit(X, y)
print(f"Linear Regression score: {reg.score(X, y)}")

regr_svm = svm.SVR()
regr_svm.fit(X_train, y_train)
y_pred = regr_svm.predict(X_test)
print('SVM Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('SVM Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('SVM Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

predictions = regr_svm.predict(X)
predictions = predictions.reshape(-1,1)
predictions = y_scaler.inverse_transform(predictions)

new_X['box office predictions'] = predictions

new_X.to_csv('box_office_revenue/data/prediction_data.csv')
