"""
Stacked AutoEncoder (SAE) Model for Movie Recommendation
Author: Ahmad Hammam
Description: PyTorch implementation of Stacked AutoEncoders for collaborative filtering
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt

class SAE(nn.Module):
    """
    Stacked AutoEncoder for Movie Recommendation System
    
    Architecture:
    Input (nb_movies) → Encoder1 (20) → Encoder2 (10) → Decoder1 (20) → Output (nb_movies)
    """
    
    def __init__(self, nb_movies: int = 1682, encoding_layers: List[int] = [20, 10]):
        """
        Initialize the Stacked AutoEncoder
        
        Args:
            nb_movies (int): Number of movies in the dataset
            encoding_layers (List[int]): List of hidden layer sizes for encoding
        """
        super(SAE, self).__init__()
        
        self.nb_movies = nb_movies
        self.encoding_layers = encoding_layers
        
        # Build encoder layers
        layers = []
        input_size = nb_movies
        
        for i, hidden_size in enumerate(encoding_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        
        # Build decoder layers (symmetric to encoder)
        decoding_layers = encoding_layers[::-1][1:] + [nb_movies]
        
        for i, hidden_size in enumerate(decoding_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        
        # Store layers
        self.fc1 = nn.Linear(nb_movies, encoding_layers[0])      # First encoder layer
        self.fc2 = nn.Linear(encoding_layers[0], encoding_layers[1])  # Second encoder layer (bottleneck)
        self.fc3 = nn.Linear(encoding_layers[1], encoding_layers[0])  # First decoder layer
        self.fc4 = nn.Linear(encoding_layers[0], nb_movies)      # Output layer
        
        # Activation function
        self.activation = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder
        
        Args:
            x (torch.Tensor): Input tensor of user ratings
            
        Returns:
            torch.Tensor: Reconstructed ratings
        """
        # Encoding phase
        x = self.activation(self.fc1(x))  # First encoding layer
        x = self.activation(self.fc2(x))  # Bottleneck layer
        
        # Decoding phase
        x = self.activation(self.fc3(x))  # First decoding layer
        x = self.fc4(x)                   # Output layer (no activation for final layer)
        
        return x
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation
        
        Args:
            x (torch.Tensor): Input tensor of user ratings
            
        Returns:
            torch.Tensor: Latent representation
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output
        
        Args:
            x (torch.Tensor): Latent representation
            
        Returns:
            torch.Tensor: Reconstructed ratings
        """
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

class SAETrainer:
    """
    Trainer class for the Stacked AutoEncoder
    """
    
    def __init__(self, model: SAE, learning_rate: float = 0.01, weight_decay: float = 0.5):
        """
        Initialize the trainer
        
        Args:
            model (SAE): The SAE model to train
            learning_rate (float): Learning rate for optimization
            weight_decay (float): Weight decay for regularization
        """
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.train_losses = []
        self.test_losses = []
        
    def train_epoch(self, training_set: torch.Tensor, nb_users: int) -> float:
        """
        Train the model for one epoch
        
        Args:
            training_set (torch.Tensor): Training data
            nb_users (int): Number of users
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        train_loss = 0.0
        s = 0.0
        
        for id_user in range(nb_users):
            # Get user's ratings
            input_ratings = training_set[id_user].unsqueeze(0)
            target = input_ratings.clone()
            
            # Only train on users who have rated at least one movie
            if torch.sum(target > 0) > 0:
                # Forward pass
                output = self.model(input_ratings)
                
                # Mask unrated movies (set to 0)
                output[target == 0] = 0
                
                # Calculate loss
                loss = self.criterion(output, target)
                
                # Mean corrector for sparse data
                mean_corrector = self.model.nb_movies / float(torch.sum(target > 0) + 1e-10)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Accumulate loss
                train_loss += np.sqrt(loss.item() * mean_corrector)
                s += 1.0
        
        return train_loss / s if s > 0 else 0.0
    
    def evaluate(self, training_set: torch.Tensor, test_set: torch.Tensor, nb_users: int) -> float:
        """
        Evaluate the model on test set
        
        Args:
            training_set (torch.Tensor): Training data (input)
            test_set (torch.Tensor): Test data (target)
            nb_users (int): Number of users
            
        Returns:
            float: Average test loss
        """
        self.model.eval()
        test_loss = 0.0
        s = 0.0
        
        with torch.no_grad():
            for id_user in range(nb_users):
                # Use training set as input, test set as target
                input_ratings = training_set[id_user].unsqueeze(0)
                target = test_set[id_user].unsqueeze(0)
                
                # Only evaluate users who have test ratings
                if torch.sum(target > 0) > 0:
                    # Forward pass
                    output = self.model(input_ratings)
                    
                    # Mask unrated movies in test set
                    output[target == 0] = 0
                    
                    # Calculate loss
                    loss = self.criterion(output, target)
                    
                    # Mean corrector for sparse data
                    mean_corrector = self.model.nb_movies / float(torch.sum(target > 0) + 1e-10)
                    
                    # Accumulate loss
                    test_loss += np.sqrt(loss.item() * mean_corrector)
                    s += 1.0
        
        return test_loss / s if s > 0 else 0.0
    
    def train(self, training_set: torch.Tensor, test_set: torch.Tensor, 
              nb_users: int, nb_epochs: int = 200, 
              evaluate_every: int = 10, verbose: bool = True) -> Tuple[List[float], List[float]]:
        """
        Train the model for multiple epochs
        
        Args:
            training_set (torch.Tensor): Training data
            test_set (torch.Tensor): Test data
            nb_users (int): Number of users
            nb_epochs (int): Number of training epochs
            evaluate_every (int): Evaluate test loss every N epochs
            verbose (bool): Print training progress
            
        Returns:
            Tuple[List[float], List[float]]: Training and test losses
        """
        print(f"Starting training for {nb_epochs} epochs...")
        print(f"Model architecture: {self.model.nb_movies} → {' → '.join(map(str, self.model.encoding_layers))} → {' → '.join(map(str, self.model.encoding_layers[::-1]))} → {self.model.nb_movies}")
        
        for epoch in range(1, nb_epochs + 1):
            # Train one epoch
            train_loss = self.train_epoch(training_set, nb_users)
            self.train_losses.append(train_loss)
            
            # Evaluate on test set
            if epoch % evaluate_every == 0 or epoch == nb_epochs:
                test_loss = self.evaluate(training_set, test_set, nb_users)
                self.test_losses.append(test_loss)
                
                if verbose:
                    print(f'Epoch: {epoch:3d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')
            elif verbose and epoch % 50 == 0:
                print(f'Epoch: {epoch:3d} | Train Loss: {train_loss:.4f}')
        
        print("Training completed!")
        return self.train_losses, self.test_losses
    
    def plot_losses(self) -> None:
        """
        Plot training and test losses
        """
        plt.figure(figsize=(12, 5))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot test loss
        if self.test_losses:
            plt.subplot(1, 2, 2)
            epochs_test = np.arange(10, len(self.train_losses) + 1, 10)
            if len(epochs_test) > len(self.test_losses):
                epochs_test = epochs_test[:len(self.test_losses)]
            plt.plot(epochs_test, self.test_losses[:len(epochs_test)], 
                    label='Test Loss', color='red', marker='o')
            plt.title('Test Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class MovieRecommender:
    """
    Movie recommendation system using trained SAE
    """
    
    def __init__(self, model: SAE, movies_data: Optional[dict] = None):
        """
        Initialize the recommender
        
        Args:
            model (SAE): Trained SAE model
            movies_data (dict, optional): Movie metadata
        """
        self.model = model
        self.movies_data = movies_data
        
    def predict_ratings(self, user_ratings: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings for all movies for a given user
        
        Args:
            user_ratings (torch.Tensor): User's known ratings
            
        Returns:
            torch.Tensor: Predicted ratings for all movies
        """
        self.model.eval()
        with torch.no_grad():
            if user_ratings.dim() == 1:
                user_ratings = user_ratings.unsqueeze(0)
            predictions = self.model(user_ratings)
        return predictions.squeeze()
    
    def recommend_movies(self, user_ratings: torch.Tensor, top_k: int = 10, 
                        exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        Recommend top-k movies for a user
        
        Args:
            user_ratings (torch.Tensor): User's known ratings
            top_k (int): Number of recommendations to return
            exclude_rated (bool): Whether to exclude already rated movies
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, predicted_rating) tuples
        """
        predictions = self.predict_ratings(user_ratings)
        
        if exclude_rated:
            # Set already rated movies to 0
            predictions[user_ratings > 0] = 0
        
        # Get top-k recommendations
        top_ratings, top_indices = torch.topk(predictions, top_k)
        
        recommendations = [(idx.item(), rating.item()) 
                          for idx, rating in zip(top_indices, top_ratings)]
        
        return recommendations
    
    def get_similar_users(self, user_ratings: torch.Tensor, 
                         all_user_ratings: torch.Tensor, top_k: int = 5) -> List[int]:
        """
        Find users with similar preferences
        
        Args:
            user_ratings (torch.Tensor): Target user's ratings
            all_user_ratings (torch.Tensor): All users' ratings
            top_k (int): Number of similar users to return
            
        Returns:
            List[int]: Indices of most similar users
        """
        # Encode user preferences to latent space
        user_encoding = self.model.encode(user_ratings.unsqueeze(0))
        all_encodings = self.model.encode(all_user_ratings)
        
        # Calculate similarities (cosine similarity)
        similarities = torch.cosine_similarity(user_encoding, all_encodings)
        
        # Get top-k similar users
        _, top_indices = torch.topk(similarities, top_k + 1)  # +1 to exclude self
        
        return top_indices[1:].tolist()  # Exclude the user themselves

def main():
    """
    Example usage of the SAE model
    """
    print("Stacked AutoEncoder for Movie Recommendation")
    print("=" * 50)
    
    # Example parameters
    nb_movies = 1682
    nb_users = 943
    
    # Initialize model
    sae = SAE(nb_movies=nb_movies, encoding_layers=[20, 10])
    print(f"Model initialized with {sum(p.numel() for p in sae.parameters())} parameters")
    
    # Initialize trainer
    trainer = SAETrainer(sae, learning_rate=0.01, weight_decay=0.5)
    
    # This would be used with actual data:
    # training_set = torch.FloatTensor(training_data)
    # test_set = torch.FloatTensor(test_data)
    # train_losses, test_losses = trainer.train(training_set, test_set, nb_users, nb_epochs=200)
    
    print("Model ready for training with actual MovieLens data!")

if __name__ == "__main__":
    main()