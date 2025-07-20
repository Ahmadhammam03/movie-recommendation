"""
Complete Training Pipeline for SAE Movie Recommendation
Author: Ahmad Hammam
Description: End-to-end training and evaluation pipeline
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

from .sae_model import SAE, SAETrainer, MovieRecommender
from .data_loader import MovieLensDataLoader

class SAEExperiment:
    """
    Complete experiment pipeline for SAE movie recommendation
    """
    
    def __init__(self, data_path: str = "data/", model_save_path: str = "models/"):
        """
        Initialize the experiment
        
        Args:
            data_path (str): Path to data directory
            model_save_path (str): Path to save models
        """
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.results = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_save_path, exist_ok=True)
        
        # Initialize components
        self.data_loader = MovieLensDataLoader(data_path)
        self.model = None
        self.trainer = None
        self.recommender = None
        
        # Data storage
        self.training_set = None
        self.test_set = None
        self.nb_users = 0
        self.nb_movies = 0
        
    def load_and_prepare_data(self, use_custom_csv: bool = True) -> None:
        """
        Load and prepare the MovieLens data
        
        Args:
            use_custom_csv (bool): Whether to use your custom CSV files
        """
        print("=" * 60)
        print("LOADING AND PREPARING DATA")
        print("=" * 60)
        
        if use_custom_csv:
            # Use your custom training_set.csv and test_set.csv files
            print("Loading custom CSV files...")
            
            try:
                # Load your training set
                training_df = pd.read_csv(os.path.join(self.data_path, "ml-1m", "training_set.csv"))
                test_df = pd.read_csv(os.path.join(self.data_path, "ml-1m", "test_set.csv"))
                
                print(f"✅ Loaded training set: {len(training_df):,} ratings")
                print(f"✅ Loaded test set: {len(test_df):,} ratings")
                
                # Convert to the format expected by SAE
                # Assuming columns: User, Movie, Rating, Timestamp
                training_array = training_df[['User', 'Movie', 'Rating', 'Timestamp']].values
                test_array = test_df[['User', 'Movie', 'Rating', 'Timestamp']].values
                
                # Get dataset info
                self.nb_users = int(max(max(training_array[:, 0]), max(test_array[:, 0])))
                self.nb_movies = int(max(max(training_array[:, 1]), max(test_array[:, 1])))
                
                print(f"📊 Dataset Statistics:")
                print(f"   - Users: {self.nb_users:,}")
                print(f"   - Movies: {self.nb_movies:,}")
                print(f"   - Training ratings: {len(training_array):,}")
                print(f"   - Test ratings: {len(test_array):,}")
                
                # Convert to user-item matrices
                training_matrix = self._convert_to_user_item_matrix(training_array)
                test_matrix = self._convert_to_user_item_matrix(test_array)
                
                # Convert to PyTorch tensors
                self.training_set = torch.FloatTensor(training_matrix)
                self.test_set = torch.FloatTensor(test_matrix)
                
                print(f"✅ Created PyTorch tensors:")
                print(f"   - Training tensor shape: {self.training_set.shape}")
                print(f"   - Test tensor shape: {self.test_set.shape}")
                
            except FileNotFoundError as e:
                print(f"❌ Error: Could not find CSV files: {e}")
                print("Please ensure training_set.csv and test_set.csv are in data/ml-1m/")
                return
                
        else:
            # Use the standard MovieLens format (if you had .dat files)
            try:
                self.data_loader.load_movielens_100k()
                self.data_loader.get_dataset_info()
                
                self.nb_users = self.data_loader.nb_users
                self.nb_movies = self.data_loader.nb_movies
                
                self.training_set, self.test_set = self.data_loader.prepare_torch_tensors()
                
            except Exception as e:
                print(f"❌ Error loading standard format: {e}")
                return
    
    def _convert_to_user_item_matrix(self, data: np.ndarray) -> List[List[float]]:
        """
        Convert rating data to user-item matrix
        
        Args:
            data (np.ndarray): Rating data [User, Movie, Rating, Timestamp]
            
        Returns:
            List[List[float]]: User-item matrix
        """
        new_data = []
        for id_user in range(1, self.nb_users + 1):
            # Get movies rated by this user
            user_mask = data[:, 0] == id_user
            id_movies = data[user_mask, 1]
            id_ratings = data[user_mask, 2]
            
            # Create rating vector for all movies
            ratings = np.zeros(self.nb_movies)
            ratings[id_movies - 1] = id_ratings  # Movie IDs are 1-indexed
            new_data.append(list(ratings))
        
        return new_data
    
    def initialize_model(self, encoding_layers: List[int] = [20, 10], 
                        learning_rate: float = 0.01, weight_decay: float = 0.5) -> None:
        """
        Initialize the SAE model and trainer
        
        Args:
            encoding_layers (List[int]): Hidden layer sizes
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
        """
        print("\n" + "=" * 60)
        print("INITIALIZING MODEL")
        print("=" * 60)
        
        if self.nb_movies == 0:
            raise ValueError("Please load data first!")
        
        # Initialize model
        self.model = SAE(nb_movies=self.nb_movies, encoding_layers=encoding_layers)
        
        # Initialize trainer
        self.trainer = SAETrainer(self.model, learning_rate=learning_rate, weight_decay=weight_decay)
        
        # Initialize recommender
        self.recommender = MovieRecommender(self.model)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Model initialized:")
        print(f"   - Architecture: {self.nb_movies} → {' → '.join(map(str, encoding_layers))} → {' → '.join(map(str, encoding_layers[::-1]))} → {self.nb_movies}")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Weight decay: {weight_decay}")
    
    def train_model(self, nb_epochs: int = 200, evaluate_every: int = 10, 
                   save_best: bool = True) -> Dict:
        """
        Train the SAE model
        
        Args:
            nb_epochs (int): Number of training epochs
            evaluate_every (int): Evaluate every N epochs
            save_best (bool): Save best model during training
            
        Returns:
            Dict: Training results
        """
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        
        if self.trainer is None:
            raise ValueError("Please initialize model first!")
        
        # Train the model
        train_losses, test_losses = self.trainer.train(
            self.training_set, self.test_set, self.nb_users,
            nb_epochs=nb_epochs, evaluate_every=evaluate_every, verbose=True
        )
        
        # Store results
        self.results['train_losses'] = train_losses
        self.results['test_losses'] = test_losses
        self.results['final_train_loss'] = train_losses[-1] if train_losses else None
        self.results['final_test_loss'] = test_losses[-1] if test_losses else None
        self.results['nb_epochs'] = nb_epochs
        
        # Save best model
        if save_best:
            self.save_model("best_sae_model.pth")
        
        print(f"\n✅ Training completed!")
        print(f"   - Final training loss: {self.results['final_train_loss']:.4f}")
        print(f"   - Final test loss: {self.results['final_test_loss']:.4f}")
        
        return self.results
    
    def evaluate_model(self) -> Dict:
        """
        Evaluate the trained model
        
        Returns:
            Dict: Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATING MODEL")
        print("=" * 60)
        
        if self.model is None:
            raise ValueError("Please train model first!")
        
        # Calculate final test loss
        final_test_loss = self.trainer.evaluate(self.training_set, self.test_set, self.nb_users)
        
        # Calculate some additional metrics
        self.model.eval()
        with torch.no_grad():
            # Sample some users for recommendation quality assessment
            sample_users = min(10, self.nb_users)
            total_mae = 0.0
            total_rmse = 0.0
            valid_predictions = 0
            
            for user_id in range(sample_users):
                user_ratings = self.training_set[user_id].unsqueeze(0)
                predicted_ratings = self.model(user_ratings).squeeze()
                actual_test_ratings = self.test_set[user_id]
                
                # Only evaluate on movies that were actually rated in test set
                mask = actual_test_ratings > 0
                if mask.sum() > 0:
                    pred_subset = predicted_ratings[mask]
                    actual_subset = actual_test_ratings[mask]
                    
                    # Calculate MAE and RMSE
                    mae = torch.mean(torch.abs(pred_subset - actual_subset))
                    rmse = torch.sqrt(torch.mean((pred_subset - actual_subset) ** 2))
                    
                    total_mae += mae.item()
                    total_rmse += rmse.item()
                    valid_predictions += 1
        
        # Store evaluation results
        evaluation_results = {
            'test_loss': final_test_loss,
            'sample_mae': total_mae / valid_predictions if valid_predictions > 0 else None,
            'sample_rmse': total_rmse / valid_predictions if valid_predictions > 0 else None,
            'evaluated_users': valid_predictions
        }
        
        self.results.update(evaluation_results)
        
        print(f"📊 Evaluation Results:")
        print(f"   - Test Loss (RMSE): {final_test_loss:.4f}")
        if evaluation_results['sample_mae']:
            print(f"   - Sample MAE: {evaluation_results['sample_mae']:.4f}")
            print(f"   - Sample RMSE: {evaluation_results['sample_rmse']:.4f}")
        
        return evaluation_results
    
    def demonstrate_recommendations(self, user_ids: List[int] = [1, 2, 5], 
                                  top_k: int = 10) -> None:
        """
        Demonstrate recommendations for sample users
        
        Args:
            user_ids (List[int]): User IDs to demonstrate
            top_k (int): Number of recommendations
        """
        print("\n" + "=" * 60)
        print("DEMONSTRATION - MOVIE RECOMMENDATIONS")
        print("=" * 60)
        
        if self.recommender is None:
            raise ValueError("Please train model first!")
        
        for user_id in user_ids:
            if user_id <= self.nb_users:
                print(f"\n🎬 Recommendations for User {user_id}:")
                print("-" * 40)
                
                # Get user's rating vector
                user_ratings = self.training_set[user_id - 1]  # Convert to 0-indexed
                
                # Get recommendations
                recommendations = self.recommender.recommend_movies(
                    user_ratings, top_k=top_k, exclude_rated=True
                )
                
                # Display recommendations
                for i, (movie_idx, predicted_rating) in enumerate(recommendations, 1):
                    movie_id = movie_idx + 1  # Convert back to 1-indexed
                    print(f"   {i:2d}. Movie ID {movie_id:4d} - Predicted Rating: {predicted_rating:.2f}")
                
                # Show user's actual rating pattern
                rated_movies = torch.nonzero(user_ratings).squeeze()
                if len(rated_movies) > 0:
                    avg_rating = user_ratings[rated_movies].mean().item()
                    print(f"   📊 User {user_id} has rated {len(rated_movies)} movies (avg: {avg_rating:.2f})")
    
    def plot_training_progress(self) -> None:
        """
        Plot training progress
        """
        if 'train_losses' not in self.results:
            print("No training results to plot!")
            return
        
        self.trainer.plot_losses()
    
    def save_model(self, filename: str) -> None:
        """
        Save the trained model
        
        Args:
            filename (str): Filename to save model
        """
        if self.model is None:
            raise ValueError("No model to save!")
        
        filepath = os.path.join(self.model_save_path, filename)
        
        # Save model state dict and metadata
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'nb_movies': self.nb_movies,
                'encoding_layers': self.model.encoding_layers
            },
            'training_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(save_dict, filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load_model(self, filename: str) -> None:
        """
        Load a saved model
        
        Args:
            filename (str): Filename of saved model
        """
        filepath = os.path.join(self.model_save_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Initialize model with saved config
        config = checkpoint['model_config']
        self.nb_movies = config['nb_movies']
        self.model = SAE(nb_movies=config['nb_movies'], encoding_layers=config['encoding_layers'])
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load results if available
        if 'training_results' in checkpoint:
            self.results = checkpoint['training_results']
        
        # Initialize recommender
        self.recommender = MovieRecommender(self.model)
        
        print(f"✅ Model loaded from: {filepath}")
    
    def save_results(self, filename: str = "experiment_results.json") -> None:
        """
        Save experiment results to JSON
        
        Args:
            filename (str): Filename for results
        """
        filepath = os.path.join(self.model_save_path, filename)
        
        # Convert numpy/torch types to Python types for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                json_results[key] = value.tolist() if hasattr(value, 'tolist') else float(value)
            elif isinstance(value, (np.floating, np.integer)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Results saved to: {filepath}")

def main():
    """
    Main function to run complete experiment
    """
    print("🎬 SAE Movie Recommendation System - Complete Experiment")
    print("=" * 80)
    
    # Initialize experiment
    experiment = SAEExperiment(data_path="data/", model_save_path="models/")
    
    try:
        # Step 1: Load and prepare data
        experiment.load_and_prepare_data(use_custom_csv=True)
        
        # Step 2: Initialize model
        experiment.initialize_model(encoding_layers=[20, 10], learning_rate=0.01, weight_decay=0.5)
        
        # Step 3: Train model
        experiment.train_model(nb_epochs=200, evaluate_every=10, save_best=True)
        
        # Step 4: Evaluate model
        experiment.evaluate_model()
        
        # Step 5: Demonstrate recommendations
        experiment.demonstrate_recommendations(user_ids=[1, 2, 5, 10], top_k=10)
        
        # Step 6: Plot results
        experiment.plot_training_progress()
        
        # Step 7: Save results
        experiment.save_results()
        
        print("\n🎉 Experiment completed successfully!")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()