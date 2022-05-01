import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

def baseline(data, query, k, distance_fn, choice_fn):
    movie_matches = []
    for index, example in enumerate(data):
        match = distance_fn(example[:-1], query)
        
        if match:
            movie_matches.append((match, index))
        
    sorted_movie_matches = sorted(movie_matches, reverse=True)
    
    n_number_movies = sorted_movie_matches[:k]
    
    k_nearest_labels = [data[i][-1] for distance, i in n_number_movies]
    
    return n_number_movies, choice_fn(k_nearest_labels)

def matches(point1, point2):
    for i in range(len(point1)):        
        if (point1[i] != point2[i]):
            return False        
    return True

def recommend_movies(data_file, movie_query, algorithm, k_recommendations):
    raw_movies_data = []
    with open(data_file, 'r', encoding='utf-8') as md:
        # Discard the first line (headings)
        next(md)

        # Read the data into memory
        for line in md.readlines():
            data_row = line.strip().split(',')
            raw_movies_data.append(data_row)

    # Prepare the data for use in the knn algorithm by picking
    # the relevant columns and converting the numeric columns
    # to numbers since they were read in as strings
    movies_recommendation_data = []
    for row in raw_movies_data:
        data_row = list(map(float, row[2:]))
        movies_recommendation_data.append(data_row)

    # Use the baseline algorithm to get the 5 movies that are most
    # similar to The Post.    
    recommendation_indices, _ = algorithm(
        movies_recommendation_data, movie_query, k=k_recommendations,
        distance_fn=matches, choice_fn=lambda x: None
    )

    movie_recommendations = []
    for _, index in recommendation_indices:
        movie_recommendations.append(raw_movies_data[index])

    return movie_recommendations

if __name__ == '__main__':
    file = "KaggleMovies/data/cleanMovies.csv"
    number_of_movies = 15
    
    movie = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # feature vector for movie
    movie_title = "The Lost City"
    recommended_movies = recommend_movies(data_file=file, movie_query=movie, algorithm=baseline, k_recommendations=number_of_movies)

    # Print recommended movie titles
    print()
    print ("Movie Recommendations for " + movie_title + ":")
    print ("Number of Genres: 2 (Adventure, Comedy)")
    for recommendation in recommended_movies:
        print(recommendation[1])
    print()
    
    #---------------------------------------------------------------------------------------------------------------
        
    movie = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # feature vector for movie
    movie_title = "Turning Red"
    recommended_movies = recommend_movies(data_file=file, movie_query=movie, algorithm=baseline, k_recommendations=number_of_movies)

    # Print recommended movie titles
    print ("Movie Recommendations for " + movie_title + ":")
    print ("Number of Genres: 3 (Adventure, Animation, Comedy)")
    for recommendation in recommended_movies:
        print(recommendation[1])
    print()
    
    #---------------------------------------------------------------------------------------------------------------
    
    movie = [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # feature vector for movie
    movie_title = "Morbius"
    recommended_movies = recommend_movies(data_file=file, movie_query=movie, algorithm=baseline, k_recommendations=number_of_movies)

    # Print recommended movie titles
    print ("Movie Recommendations for " + movie_title + ":")
    print ("Number of Genres: 4 (Adventure, Drama, Action, Sci-Fi)")
    for recommendation in recommended_movies:
        print(recommendation[1])
    print()
    
    #---------------------------------------------------------------------------------------------------------------
    
    movie = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] # feature vector for movie
    movie_title = "Encanto"
    recommended_movies = recommend_movies(data_file=file, movie_query=movie, algorithm=baseline, k_recommendations=number_of_movies)

    # Print recommended movie titles
    print ("Movie Recommendations for " + movie_title + ":")
    print ("Number of Genres: 5 (Adventure, Animation, Children, Comedy, Musical)")
    for recommendation in recommended_movies:
        print(recommendation[1])
    print()
    
    # Plot the genres graph
    genres = ('Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama',
            'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'War', 'Musical',
            'Documentary', 'IMAX', 'Western', 'Film-Noir')
																			
    y_pos = np.arange(len(genres))
    total_count = [1263,611,664,3756,779,1596,4361,1828,1199,1894,978,573,980,382,334,440,158,167,87]

    plt.barh(y_pos, total_count, align='center', alpha=0.5)
    plt.yticks(y_pos, genres)
    plt.xlabel('Total')
    plt.title('Total Count per Movie Genre')

    plt.show()