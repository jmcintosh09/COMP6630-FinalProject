import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

genres = ('Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama',
          'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'War', 'Musical',
          'Documentary', 'IMAX', 'Western', 'Film-Noir')
																			

y_pos = np.arange(len(genres))
total_count = [1263,611,664,3756,779,1596,4361,1828,1199,1894,978,573,980,382,334,440,158,167,87]

plt.barh(y_pos, total_count, align='center', alpha=0.5)
plt.yticks(y_pos, genres)
plt.xlabel('Total')
plt.title('Movie Genres')

plt.show()