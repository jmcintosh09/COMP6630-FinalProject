from turtle import title
import pandas as pd
import csv
from sklearn import datasets

csv_file = "movies.csv"
csv_list = []

# Read to list
with open(csv_file, mode="r", encoding='utf-8') as fp:
    csvreader = csv.reader(fp)
    
    print ()
    fields = next(csvreader)
    
    
    for row in csvreader:
        csv_list.append(row)

# For each line we save the key:values to a dict
pandas_label_list = []
pandas_feature_list = []
for line in csv_list:
    feature_dict = {}

    ##items = line.split(sep="),")
    ##title, genres = line.split("),")
    title = line[0]
    genres = line[-1]
    pandas_label_list.append({'Title': title})
    feature_dict['Title'] = title
    genres = genres.split("|")

    for genre in genres:
        ##feature_name, count = pair.split(':')        
        ##genres = pair.split('|')
        feature_dict[genre] = int(1)

    pandas_feature_list.append(feature_dict)
    
X = pd.DataFrame(pandas_feature_list).fillna(0)
Y = pd.DataFrame(pandas_label_list)

#print (X)
#print (Y)
X.to_csv("movies_clean.csv")
#Y.to_csv("a4a_Y.csv")
