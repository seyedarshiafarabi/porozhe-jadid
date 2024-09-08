import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
data = pd.merge(ratings, movies, on='movieId')

max_user_value = ratings["userId"].max()
movie_numbers = movies.shape[0]

def merge_memmap_matrices(file1, file2, result_file, dtype, axis=0):
    """
    Merges two memory-mapped matrices along a specified axis.

    Parameters:
    - file1: str, path to the first memory-mapped file.
    - file2: str, path to the second memory-mapped file.
    - result_file: str, path to the resulting memory-mapped file.
    - dtype: data type of the matrices (e.g., 'float32').
    - axis: int, axis along which to concatenate the matrices. Default is 0.

    Returns:
    - result_mmap: memory-mapped array containing the merged result.
    """
    # Open the memory-mapped matrices
    mmap1 = np.memmap(file1, dtype=dtype, mode='r', shape=(max_user_value, movie_numbers))
    mmap2 = np.memmap(file2, dtype=dtype, mode='r', shape=(max_user_value, max_user_value))
    
    # Reshape the memory-mapped arrays to their correct dimensions
    shape1 = mmap1.shape
    shape2 = mmap2.shape
    
    if axis == 0:
        result_shape = (shape1[0] + shape2[0], shape1[1])
    elif axis == 1:
        result_shape = (shape1[0], shape1[1] + shape2[1])
    else:
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")
    
    # Create the memory-mapped file for the result
    result_mmap = np.memmap(result_file, dtype=dtype, mode='w+', shape=result_shape)
    
    # Perform the concatenation
    if axis == 0:
        result_mmap[:shape1[0], :] = mmap1
        result_mmap[shape1[0]:, :] = mmap2
    elif axis == 1:
        result_mmap[:, :shape1[1]] = mmap1
        result_mmap[:, shape1[1]:] = mmap2
    
    # Ensure data is written to disk
    result_mmap.flush()
    
    return result_mmap

def merge_memmap_matrices_trans(file1, file2, result_file, dtype, axis=0):
    """
    Merges two memory-mapped matrices along a specified axis.

    Parameters:
    - file1: str, path to the first memory-mapped file.
    - file2: str, path to the second memory-mapped file.
    - result_file: str, path to the resulting memory-mapped file.
    - dtype: data type of the matrices (e.g., 'float32').
    - axis: int, axis along which to concatenate the matrices. Default is 0.

    Returns:
    - result_mmap: memory-mapped array containing the merged result.
    """
    # Open the memory-mapped matrices
    mmap1 = np.memmap(file1, dtype=dtype, mode='r', shape=(movie_numbers, movie_numbers))
    mmap2 = np.memmap(file2, dtype=dtype, mode='r', shape=(movie_numbers, max_user_value))
    mmap2 = mmap2.transpose()
    mmap2= np.transpose(mmap2)
    # Reshape the memory-mapped arrays to their correct dimensions
    shape1 = mmap1.shape
    shape2 = mmap2.shape
    
    if axis == 0:
        result_shape = (shape1[0] + shape2[0], shape1[1])
    elif axis == 1:
        result_shape = (shape1[0], shape1[1] + shape2[1])
    else:
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")
    
    # Create the memory-mapped file for the result
    result_mmap = np.memmap(result_file, dtype=dtype, mode='w+', shape=result_shape)
    
    # Perform the concatenation
    if axis == 0:
        result_mmap[:shape1[0], :] = mmap1
        result_mmap[shape1[0]:, :] = mmap2
    elif axis == 1:
        result_mmap[:, :shape1[1]] = mmap1
        result_mmap[:, shape1[1]:] = mmap2
    
    # Ensure data is written to disk
    result_mmap.flush()
    
    return result_mmap

def csv_to_memmap(csv_file, mmap_file, dtype='float16', delimiter=','):
    """
    Imports a CSV file into a memory-mapped file.

    Parameters:
    - csv_file: str, path to the CSV file.
    - mmap_file: str, path where the memory-mapped file will be stored.
    - dtype: data type for the memory-mapped file (default is 'float32').
    - delimiter: str, delimiter used in the CSV file (default is ',').

    Returns:
    - mmap: memory-mapped array containing the CSV data.
    """
    # Load the CSV data into a NumPy array
    data = np.loadtxt(csv_file, delimiter=delimiter, dtype=dtype)

    # Create a memory-mapped file with the same shape as the loaded data
    mmap = np.memmap(mmap_file, dtype=dtype, mode='w+', shape=data.shape)

    # Copy the data to the memory-mapped file
    mmap[:] = data[:]

    # Ensure the data is written to disk
    mmap.flush()

    return mmap

def ratings_csv_to_user_item_matrix(csv_file, output_file, dtype='float16'):
    """
    Converts a MovieLens ratings CSV file into a user-item matrix and saves it as a memory-mapped file.

    Parameters:
    - csv_file: str, path to the ratings CSV file.
    - output_file: str, path where the memory-mapped user-item matrix will be stored.
    - dtype: data type for the matrix (default is 'float32').

    Returns:
    - user_item_matrix: memory-mapped array representing the user-item matrix.
    """
    # Load the CSV data into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Create a pivot table to convert the DataFrame into a user-item matrix
    user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Convert the DataFrame to a NumPy array
    matrix = user_item_matrix.to_numpy(dtype=dtype)
    
    # Create a memory-mapped file for the matrix
    mmap = np.memmap(output_file, dtype=dtype, mode='w+', shape=matrix.shape)
    
    # Copy the data to the memory-mapped file
    mmap[:] = matrix[:]
    
    # Ensure the data is written to disk
    mmap.flush()

    return mmap

def item_item_similarity_matrix(user_item_mmap_file, output_file, dtype='float16'):
    """
    Computes the item-item similarity matrix from a user-item memory-mapped file
    and saves it as a memory-mapped file.

    Parameters:
    - user_item_mmap_file: str, path to the memory-mapped user-item matrix file.
    - output_file: str, path where the memory-mapped item-item similarity matrix will be stored.
    - dtype: data type for the matrix (default is 'float32').

    Returns:
    - similarity_mmap: memory-mapped array representing the item-item similarity matrix.
    """
    # Load the user-item matrix
    user_item_matrix = np.memmap(user_item_mmap_file, dtype=dtype, shape=(max_user_value, movie_numbers), mode='r')
    
    # Reshape the matrix (assuming it was saved as a 2D matrix)
    num_users, num_items = user_item_matrix.shape
    
    # Compute the cosine similarity between items (transpose the matrix for item-based similarity)
    similarity_matrix = cosine_similarity(user_item_matrix.T)
    
    # Create a memory-mapped file for the similarity matrix
    similarity_mmap = np.memmap(output_file, dtype=dtype, mode='w+', shape=similarity_matrix.shape)
    
    # Copy the similarity data to the memory-mapped file
    similarity_mmap[:] = similarity_matrix[:]
    
    # Ensure the data is written to disk
    similarity_mmap.flush()

    return similarity_mmap

def user_user_similarity_matrix(user_item_mmap_file, output_file, dtype='float16'):
    """
    Computes the user-user similarity matrix from a user-item memory-mapped file
    and saves it as a memory-mapped file.

    Parameters:
    - user_item_mmap_file: str, path to the memory-mapped user-item matrix file.
    - output_file: str, path where the memory-mapped user-user similarity matrix will be stored.
    - dtype: data type for the matrix (default is 'float32').

    Returns:
    - similarity_mmap: memory-mapped array representing the user-user similarity matrix.
    """
    # Load the user-item matrix
    user_item_matrix = np.memmap(user_item_mmap_file, dtype=dtype, shape=(max_user_value, movie_numbers),mode='r')
    
    # Reshape the matrix (assuming it was saved as a 2D matrix)
    num_users, num_items = user_item_matrix.shape
    
    # Compute the cosine similarity between users
    similarity_matrix = cosine_similarity(user_item_matrix)
    
    # Create a memory-mapped file for the similarity matrix
    similarity_mmap = np.memmap(output_file, dtype=dtype, mode='w+', shape=similarity_matrix.shape)
    
    # Copy the similarity data to the memory-mapped file
    similarity_mmap[:] = similarity_matrix[:]
    
    # Ensure the data is written to disk
    similarity_mmap.flush()

    return similarity_mmap

# Example usage
csv_file = 'data.csv'
mmap_file = 'data_mmap.dat'
dtype = 'float16'  # Change this if your data has a different type

# Convert the CSV to a memory-mapped file
user_item_mmap_file= ratings_csv_to_user_item_matrix('ratings.csv', 'ratings.array',dtype)
userXuser = user_user_similarity_matrix('ratings.array', 'userXuser.array', dtype)
itemXitem = item_item_similarity_matrix('ratings.array', 'itemXitem.array', dtype)

file1 = 'ratings.array'
file2 = 'userXuser.array'
file3 = 'itemXitem.array'
result_file = 'merged_matrix.dat'
result_file2 = 'merged_matrix2.dat'
result_file3 = 'merged_matrix3.dat'
dtype = 'float16'  # Change this to match the data type of your matrices

# Merge the matrices along rows (axis=0)
merged_mmap1 = merge_memmap_matrices(file1, file2, result_file, dtype, axis=1)
merged_mmap2 = merge_memmap_matrices_trans(file3, file1, result_file2, dtype, axis=1)
merged_mmap3 = merge_memmap_matrices(result_file, result_file, result_file, dtype, axis=0)
# The result is now stored in 'merged_matrix.dat'
print("Merged matrix shape:", merged_mmap1.shape)
print("hi")