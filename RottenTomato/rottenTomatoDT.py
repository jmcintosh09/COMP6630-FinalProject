import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

dataset= pd.read_csv("RottenTomato/data/cleanRottenTomatoesMovies.csv")
#getting only columns pertaining to tomatometer
feature_columns=["content_rating_as_int","tomatometer_rating","tomatometer_count","audience_rating","audience_count","tomatometer_top_critics_count","tomatometer_fresh_critics_count","tomatometer_rotten_critics_count"]
X=dataset[feature_columns]

y=dataset.tomatometer_status_results
#spliting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.70, random_state=42)
#creating and fitting the decision tree
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#creating a new dataset to be used in movie recommendation machine that predicts all movies
feature_columns_all_movies=["movie_title","genres","content_rating_as_int","tomatometer_rating","tomatometer_count","audience_rating","audience_count","tomatometer_top_critics_count","tomatometer_fresh_critics_count","tomatometer_rotten_critics_count"]
all_movies_X=dataset[feature_columns_all_movies]
#create temp dataset without movie titles and genres
all_movies_X_drop=all_movies_X.drop(['movie_title','genres'],axis=1)

y_pred2 = clf.predict(all_movies_X_drop)

#get the number of movies
R,F=all_movies_X_drop.shape

# create string results for the predicted values
resultsString=[]
for z in range(R):
    if y_pred2[z]==1:
        resultsString.append("Certified Fresh")
    elif y_pred2[z]==0:
        resultsString.append("Fresh")
    else:
        resultsString.append("Rotten")


#adding the y_predict values to dataframe
all_movies_X['predicted_y']=y_pred2

#adding y_predict String values to dataframe
all_movies_X['predicted_y_string']=resultsString

#removing columns that will not be needed for movie recommendation machine
FinalResults=all_movies_X.drop(["content_rating_as_int","tomatometer_count","audience_rating","audience_count","tomatometer_top_critics_count","tomatometer_fresh_critics_count","tomatometer_rotten_critics_count"], axis=1)

FinalResults.to_csv('RottenTomato/data/RottenTomatoesMoviesForMachine.csv', index=False) 
