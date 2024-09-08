import numpy as np
import pandas as pd
import gc

# Load data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
data = pd.merge(ratings, movies, on='movieId')

# Get max values
max_user_value = ratings["userId"].max()
max_movie_value = ratings["movieId"].max()

# Initialize memory-mapped arrays
_userxitem_mem = np.memmap('a.array', dtype=np.float16, mode='w+', shape=(max_user_value, max_movie_value))
_userxuser_mem = np.memmap('b.array', dtype=np.float16, mode='w+', shape=(max_user_value, max_user_value))
_itemxitem_mem = np.memmap('_itemxitem.dat', dtype=np.float16, mode='w+', shape=(max_movie_value, max_movie_value))
_itemxuser_mem = _userxitem_mem.transpose()

def concatenate_and_clear(array1, array2):
    result = np.concatenate((array1, array2), axis=1)
    del array1
    del array2
    gc.collect()
    return result

A = concatenate_and_clear(_userxitem_mem,_userxuser_mem)
B = concatenate_and_clear(_itemxitem_mem,_itemxuser_mem)
C = np.concatenate((A , B) , axis=0)

del A
del B