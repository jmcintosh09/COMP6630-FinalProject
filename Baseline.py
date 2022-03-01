from collections import Counter
import math

def knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []
    
    # 3. For each example in the data
    for index, example in enumerate(data):
        # 3.1 Calculate the distance between the query example and the current
        # example from the data.
        distance = distance_fn(example[:-1], query)
        
        # 3.2 Add the distance and the index of the example to an ordered collection
        neighbor_distances_and_indices.append((distance, index))
    
    # 4. Sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    
    # 5. Pick the first K entries from the sorted collection
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    
    # 6. Get the labels of the selected K entries
    k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_and_indices]

    # 7. If regression (choice_fn = mean), return the average of the K labels
    # 8. If classification (choice_fn = mode), return the mode of the K labels
    return k_nearest_distances_and_indices , choice_fn(k_nearest_labels)

def baseline(data, query, k, distance_fn, choice_fn):
    movie_matches = []
    for index, example in enumerate(data):
        count = distance_fn(example[:-1], query)
        
        movie_matches.append((count, index))
        
    sorted_movie_matches = sorted(movie_matches, reverse=True)
    
    n_number_movies = sorted_movie_matches[:k]
    
    k_nearest_labels = [data[i][-1] for distance, i in n_number_movies]
    
    return n_number_movies, choice_fn(k_nearest_labels)


def matches(point1, point2):
    count = 0
    for i in range(len(point1)):
        if (point1[i] > 0.0 and point2[i] > 0.0 and point1[i] == point2[i]):
            count +=1
    return count

def mean(labels):
    return sum(labels) / len(labels)

def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)

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
    file = 'movies_clean.csv'
    encanto = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] # feature vector for Encanto
    movie_title = "Encanto"
    number_of_movies = 10
    recommended_movies = recommend_movies(data_file=file, movie_query=encanto, algorithm=baseline, k_recommendations=number_of_movies)

    # Print recommended movie titles
    print ("Movie Recommendations for " + movie_title + ":")
    for recommendation in recommended_movies:
        print(recommendation[1])