import pandas as pd
from pathlib import Path  

dataset=pd.read_csv("RottenTomato/data/rotten_tomatoes_movies.csv")

dataset=dataset.fillna(0)

#changing content rating datatype from string to int
content_rating=dataset.content_rating
content_rating_int=[]
for rating in content_rating:
    if rating == "NR":
        content_rating_int.append(0)
    elif rating == "R":
        content_rating_int.append(-2)
    elif rating == "PG-13":
        content_rating_int.append(-1)
    elif rating == "PG":
        content_rating_int.append(1)
    else:
        content_rating_int.append(2)
        
dataset["content_rating_as_int"]= content_rating_int
#changing tomatomenter status from string to int 
tomatometer_status=dataset.tomatometer_status
tomatometer_status_result=[]
for status in tomatometer_status:
    if status=="Rotten":
        tomatometer_status_result.append(-1)
    elif status== "Fresh":
        tomatometer_status_result.append(0)
    else:
        tomatometer_status_result.append(1)
        
dataset["tomatometer_status_results"]= tomatometer_status_result

cleanRottenTomatoesMovies=dataset
#saving new csv
filepath = Path('RottenTomato/data/cleanRottenTomatoesMovies.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)   
cleanRottenTomatoesMovies.to_csv(filepath)


