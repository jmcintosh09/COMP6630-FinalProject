# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:51:25 2022

@author: omard
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class Classifier():

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = None
        self.combined = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.secondary = None
        self.__load_dataset()

    # Reads the dataset file into a pandas dataframe
    def __load_dataset(self):
        self.dataset = pd.read_csv(self.dataset_path)
        self.dataset.drop('Unnamed: 0', axis = 1, inplace=True)
        
        # Change the -1 labels to 0 labels
        #self.dataset['label'].replace(to_replace = -1, value = 0, inplace=True)# Comment this line out to keep the -1 labels


    # Dataset preprocessing
    def preprocess(self):
        # Seperate data and labels
        X = self.combined.drop('Title', axis = 1)
        y = self.combined['Title']

        # Train and test splitting of data with an 90% training and 10% testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.1)
        
    def load_second_dataset(self, path):
        self.secondary = pd.read_csv(path)
        self.secondary.drop(['genres','tomatometer_rating','predicted_y','predicted_y_string'], axis = 1, inplace=True)
        self.combined = pd.concat([self.dataset,self.secondary],axis=0,ignore_index=False,copy = True)
        
        
    # KNN
    def knn(self,K):
        # Create the object for the model
        knc = KNeighborsClassifier()

        # Fit the model using X as training data and y as target values
        knc.fit(self.X_train, self.y_train)

        # Predict the class labels for the provided data
        pred_knc = knc.kneighbors(self.X_test,K,False)
        return pred_knc
    
    
    
obj = Classifier('KaggleMovies\data\cleanMovies.csv')

obj.load_second_dataset('RottenTomato\data\RottenTomatoesMoviesForMachine.csv')
# Preprocess the dataset
obj.preprocess()


points = obj.knn(5)

movie = obj.y_test.to_list()
recomendations = [[obj.y_train.iat[movieID] for movieID in movieIDList] for movieIDList in points]






