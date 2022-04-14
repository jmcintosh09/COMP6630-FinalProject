import json
import pandas as pd
import numpy as np


metadata = pd.read_csv('box_office_revenue/data/movies_metadata.csv')
data = pd.read_csv('box_office_revenue/data/boxoffice.csv')

metadata = metadata.loc[:, metadata.columns.intersection(['genres', 'title', 'runtime'])]
df = metadata.merge(data, on='title')

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

X.to_csv('box_office_revenue/data/X_data.csv')
y.to_csv('box_office_revenue/data/y_data.csv')
