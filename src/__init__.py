"""
Movie Recommendation System using Stacked AutoEncoders
Author: Ahmad Hammam
Version: 1.0.0

This package provides tools for movie recommendation using:
- Stacked AutoEncoders (SAE) for collaborative filtering
- PyTorch implementation with custom training loops
- MovieLens dataset handling and preprocessing
"""

__version__ = "1.0.0"
__author__ = "Ahmad Hammam"
__description__ = "Movie Recommendation System using Stacked AutoEncoders"

# Import main classes
try:
    from .data_loader import MovieLensDataLoader
    from .sae_model import SAE, SAETrainer, MovieRecommender
    from .trainer import SAEExperiment
    from .recommender import RecommendationEngine
except ImportError:
    # Handle relative imports when running as script
    pass

__all__ = [
    'MovieLensDataLoader',
    'SAE', 
    'SAETrainer',
    'MovieRecommender',
    'SAEExperiment',
    'RecommendationEngine'
]