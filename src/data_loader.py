"""
Data Loader Module for MovieLens Dataset
Author: Ahmad Hammam
Description: Handles loading and preprocessing of MovieLens data for SAE training
"""

import numpy as np
import pandas as pd
import torch
import os
import requests
import zipfile
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class MovieLensDataLoader:
    """
    Data loader for MovieLens datasets
    """
    
    def __init__(self, data_path: str = "data/"):
        """
        Initialize the data loader
        
        Args:
            data_path (str): Path to data directory
        """
        self.data_path = data_path
        self.movies_1m = None
        self.users_1m = None
        self.ratings_1m = None
        self.training_set = None
        self.test_set = None
        self.nb_users = 0
        self.nb_movies = 0
        
    def download_datasets(self) -> None:
        """
        Download MovieLens datasets if they don't exist
        """
        # URLs for MovieLens datasets
        ml_1m_url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        ml_100k_url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_path, exist_ok=True)
        
        # Download ML-1M if not exists
        if not os.path.exists(os.path.join(self.data_path, "ml-1m")):
            print("Downloading MovieLens 1M dataset...")
            self._download_and_extract(ml_1m_url, "ml-1m.zip")
        
        # Download ML-100K if not exists
        if not os.path.exists(os.path.join(self.data_path, "ml-100k")):
            print("Downloading MovieLens 100K dataset...")
            self._download_and_extract(ml_100k_url, "ml-100k.zip")
    
    def _download_and_extract(self, url: str, filename: str) -> None:
        """
        Download and extract a zip file
        
        Args:
            url (str): Download URL
            filename (str): Local filename
        """
        filepath = os.path.join(self.data_path, filename)
        
        # Download file
        response = requests.get(url)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        # Extract zip file
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(self.data_path)
        
        # Remove zip file
        os.remove(filepath)
        print(f"Downloaded and extracted {filename}")
    
    def load_movielens_1m(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load MovieLens 1M dataset
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Movies, users, and ratings dataframes
        """
        ml_1m_path = os.path.join(self.data_path, "ml-1m")
        
        # Load movies data
        movies_columns = ['MovieID', 'Title', 'Genres']
        self.movies_1m = pd.read_csv(
            os.path.join(ml_1m_path, 'movies.dat'),
            sep='::',
            header=None,
            names=movies_columns,
            engine='python',
            encoding='latin-1'
        )
        
        # Load users data
        users_columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
        self.users_1m = pd.read_csv(
            os.path.join(ml_1m_path, 'users.dat'),
            sep='::',
            header=None,
            names=users_columns,
            engine='python',
            encoding='latin-1'
        )
        
        # Load ratings data
        ratings_columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        self.ratings_1m = pd.read_csv(
            os.path.join(ml_1m_path, 'ratings.dat'),
            sep='::',
            header=None,
            names=ratings_columns,
            engine='python',
            encoding='latin-1'
        )
        
        print(f"Loaded MovieLens 1M dataset:")
        print(f"  - Movies: {len(self.movies_1m):,}")
        print(f"  - Users: {len(self.users_1m):,}")
        print(f"  - Ratings: {len(self.ratings_1m):,}")
        
        return self.movies_1m, self.users_1m, self.ratings_1m
    
    def load_movielens_100k(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load MovieLens 100K dataset (training and test sets)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Training and test sets
        """
        ml_100k_path = os.path.join(self.data_path, "ml-100k")
        
        # Load training set
        training_set = pd.read_csv(
            os.path.join(ml_100k_path, 'u1.base'),
            delimiter='\t',
            header=None,
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )
        self.training_set = np.array(training_set, dtype='int')
        
        # Load test set
        test_set = pd.read_csv(
            os.path.join(ml_100k_path, 'u1.test'),
            delimiter='\t',
            header=None,
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )
        self.test_set = np.array(test_set, dtype='int')
        
        print(f"Loaded MovieLens 100K dataset:")
        print(f"  - Training ratings: {len(self.training_set):,}")
        print(f"  - Test ratings: {len(self.test_set):,}")
        
        return self.training_set, self.test_set
    
    def get_dataset_info(self) -> None:
        """
        Get information about users and movies
        """
        if self.training_set is None or self.test_set is None:
            raise ValueError("Please load the 100K dataset first")
        
        self.nb_users = int(max(max(self.training_set[:, 0]), max(self.test_set[:, 0])))
        self.nb_movies = int(max(max(self.training_set[:, 1]), max(self.test_set[:, 1])))
        
        print(f"Dataset Information:")
        print(f"  - Number of users: {self.nb_users}")
        print(f"  - Number of movies: {self.nb_movies}")
        print(f"  - Sparsity: {1 - (len(self.training_set) + len(self.test_set)) / (self.nb_users * self.nb_movies):.4f}")
    
    def convert_to_user_item_matrix(self, data: np.ndarray) -> List[List[float]]:
        """
        Convert data into a user-item matrix format
        
        Args:
            data (np.ndarray): Rating data in [UserID, MovieID, Rating, Timestamp] format
            
        Returns:
            List[List[float]]: User-item matrix where rows are users and columns are movies
        """
        if self.nb_users == 0 or self.nb_movies == 0:
            self.get_dataset_info()
        
        new_data = []
        for id_user in range(1, self.nb_users + 1):
            # Get movies rated by this user
            id_movies = data[:, 1][data[:, 0] == id_user]
            # Get ratings given by this user
            id_ratings = data[:, 2][data[:, 0] == id_user]
            
            # Create rating vector for all movies
            ratings = np.zeros(self.nb_movies)
            ratings[id_movies - 1] = id_ratings  # Movie IDs are 1-indexed
            new_data.append(list(ratings))
        
        return new_data
    
    def prepare_torch_tensors(self) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Prepare PyTorch tensors for training
        
        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Training and test tensors
        """
        # Convert to user-item matrices
        training_matrix = self.convert_to_user_item_matrix(self.training_set)
        test_matrix = self.convert_to_user_item_matrix(self.test_set)
        
        # Convert to PyTorch tensors
        training_tensor = torch.FloatTensor(training_matrix)
        test_tensor = torch.FloatTensor(test_matrix)
        
        print(f"Created PyTorch tensors:")
        print(f"  - Training set shape: {training_tensor.shape}")
        print(f"  - Test set shape: {test_tensor.shape}")
        
        return training_tensor, test_tensor
    
    def explore_data(self) -> None:
        """
        Explore and visualize the dataset
        """
        if self.ratings_1m is None:
            print("Please load the 1M dataset first for exploration")
            return
        
        print("\n" + "="*50)
        print("DATASET EXPLORATION")
        print("="*50)
        
        # Basic statistics
        print(f"\nRating Statistics:")
        print(self.ratings_1m['Rating'].describe())
        
        # Rating distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        self.ratings_1m['Rating'].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        
        # Users and movies distribution
        plt.subplot(1, 3, 2)
        user_ratings = self.ratings_1m['UserID'].value_counts()
        plt.hist(user_ratings, bins=50, color='lightgreen', alpha=0.7)
        plt.title('Number of Ratings per User')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Number of Users')
        
        plt.subplot(1, 3, 3)
        movie_ratings = self.ratings_1m['MovieID'].value_counts()
        plt.hist(movie_ratings, bins=50, color='lightcoral', alpha=0.7)
        plt.title('Number of Ratings per Movie')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Number of Movies')
        
        plt.tight_layout()
        plt.show()
        
        # Genre analysis
        if self.movies_1m is not None:
            self._analyze_genres()
    
    def _analyze_genres(self) -> None:
        """
        Analyze movie genres
        """
        print(f"\nGenre Analysis:")
        
        # Extract all genres
        all_genres = []
        for genres_str in self.movies_1m['Genres']:
            genres = genres_str.split('|')
            all_genres.extend(genres)
        
        # Count genres
        genre_counts = pd.Series(all_genres).value_counts()
        
        plt.figure(figsize=(12, 6))
        genre_counts.head(15).plot(kind='bar', color='purple', alpha=0.7)
        plt.title('Top 15 Movie Genres')
        plt.xlabel('Genre')
        plt.ylabel('Number of Movies')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print(f"Total unique genres: {len(genre_counts)}")
        print(f"Most common genres:")
        for genre, count in genre_counts.head(10).items():
            print(f"  {genre}: {count}")
    
    def get_movie_info(self, movie_id: int) -> Optional[Dict]:
        """
        Get information about a specific movie
        
        Args:
            movie_id (int): Movie ID
            
        Returns:
            Optional[Dict]: Movie information or None if not found
        """
        if self.movies_1m is None:
            return None
        
        movie_info = self.movies_1m[self.movies_1m['MovieID'] == movie_id]
        
        if len(movie_info) == 0:
            return None
        
        movie_row = movie_info.iloc[0]
        return {
            'id': movie_row['MovieID'],
            'title': movie_row['Title'],
            'genres': movie_row['Genres'].split('|')
        }
    
    def get_user_stats(self, user_id: int) -> Optional[Dict]:
        """
        Get statistics about a specific user
        
        Args:
            user_id (int): User ID
            
        Returns:
            Optional[Dict]: User statistics or None if not found
        """
        if self.ratings_1m is None or self.users_1m is None:
            return None
        
        user_ratings = self.ratings_1m[self.ratings_1m['UserID'] == user_id]
        user_info = self.users_1m[self.users_1m['UserID'] == user_id]
        
        if len(user_info) == 0:
            return None
        
        user_row = user_info.iloc[0]
        
        return {
            'id': user_row['UserID'],
            'gender': user_row['Gender'],
            'age': user_row['Age'],
            'occupation': user_row['Occupation'],
            'num_ratings': len(user_ratings),
            'avg_rating': user_ratings['Rating'].mean(),
            'rating_std': user_ratings['Rating'].std()
        }

def main():
    """
    Example usage of the MovieLensDataLoader
    """
    print("MovieLens Data Loader Example")
    print("=" * 40)
    
    # Initialize data loader
    loader = MovieLensDataLoader("data/")
    
    # Note: In practice, you would download the datasets
    # loader.download_datasets()
    
    # Load datasets
    try:
        # Load MovieLens 1M for exploration
        movies, users, ratings = loader.load_movielens_1m()
        
        # Load MovieLens 100K for training
        training_set, test_set = loader.load_movielens_100k()
        
        # Get dataset information
        loader.get_dataset_info()
        
        # Prepare tensors for PyTorch
        training_tensor, test_tensor = loader.prepare_torch_tensors()
        
        # Explore the data
        loader.explore_data()
        
        print("\nData loading completed successfully!")
        
    except FileNotFoundError:
        print("Datasets not found. Please download them first using:")
        print("loader.download_datasets()")

if __name__ == "__main__":
    main()