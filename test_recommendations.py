from src.trainer import SAEExperiment
import torch

# Load your trained model
experiment = SAEExperiment(data_path="data/", model_save_path="models/")
experiment.load_model("best_sae_model.pth")

# Load data for testing
experiment.load_and_prepare_data(use_custom_csv=True)

# Test recommendations for User 1
user_1_ratings = experiment.training_set[0]  # User 1 (0-indexed)
recommendations = experiment.recommender.recommend_movies(user_1_ratings, top_k=10)

print("🎬 Top 10 Movie Recommendations for User 1:")
for i, (movie_idx, rating) in enumerate(recommendations, 1):
    movie_id = movie_idx + 1  # Convert to 1-indexed
    print(f"  {i:2d}. Movie ID {movie_id:4d} - Predicted Rating: {rating:.2f}")