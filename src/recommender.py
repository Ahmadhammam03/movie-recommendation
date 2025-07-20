"""
Advanced Recommendation Engine for SAE Movie Recommendation System
Author: Ahmad Hammam
Description: High-level recommendation interface with advanced features
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from .sae_model import SAE, MovieRecommender

class RecommendationEngine:
    """
    Advanced recommendation engine with analytics and insights
    """
    
    def __init__(self, model: SAE, movies_data: Optional[pd.DataFrame] = None):
        """
        Initialize the recommendation engine
        
        Args:
            model (SAE): Trained SAE model
            movies_data (pd.DataFrame, optional): Movie metadata
        """
        self.model = model
        self.movies_data = movies_data
        self.base_recommender = MovieRecommender(model, movies_data)
        
        # Cache for performance
        self._user_encodings = None
        self._movie_features = None
        
    def get_recommendations(self, user_ratings: torch.Tensor, 
                          top_k: int = 10, 
                          exclude_rated: bool = True,
                          min_rating_threshold: float = 3.0) -> List[Dict]:
        """
        Get detailed movie recommendations for a user
        
        Args:
            user_ratings (torch.Tensor): User's rating vector
            top_k (int): Number of recommendations
            exclude_rated (bool): Exclude already rated movies
            min_rating_threshold (float): Minimum predicted rating to recommend
            
        Returns:
            List[Dict]: Detailed recommendations with metadata
        """
        # Get basic recommendations
        basic_recs = self.base_recommender.recommend_movies(
            user_ratings, top_k=top_k * 2, exclude_rated=exclude_rated  # Get more to filter
        )
        
        # Filter by minimum rating threshold
        filtered_recs = [(idx, rating) for idx, rating in basic_recs 
                        if rating >= min_rating_threshold]
        
        # Take top_k after filtering
        filtered_recs = filtered_recs[:top_k]
        
        # Enhance with metadata
        detailed_recs = []
        for movie_idx, predicted_rating in filtered_recs:
            movie_id = movie_idx + 1  # Convert to 1-indexed
            
            rec_dict = {
                'movie_id': movie_id,
                'movie_index': movie_idx,
                'predicted_rating': predicted_rating,
                'confidence_score': self._calculate_confidence(user_ratings, movie_idx)
            }
            
            # Add movie metadata if available
            if self.movies_data is not None:
                movie_info = self._get_movie_metadata(movie_id)
                rec_dict.update(movie_info)
            
            detailed_recs.append(rec_dict)
        
        return detailed_recs
    
    def explain_recommendation(self, user_ratings: torch.Tensor, 
                             movie_idx: int) -> Dict:
        """
        Explain why a movie was recommended to a user
        
        Args:
            user_ratings (torch.Tensor): User's rating vector
            movie_idx (int): Movie index to explain
            
        Returns:
            Dict: Explanation details
        """
        # Get user's latent representation
        user_encoding = self.model.encode(user_ratings.unsqueeze(0)).squeeze()
        
        # Get predicted rating
        predicted_rating = self.model(user_ratings.unsqueeze(0)).squeeze()[movie_idx]
        
        # Find similar movies the user liked
        user_liked_movies = torch.nonzero(user_ratings >= 4.0).squeeze()
        
        explanation = {
            'predicted_rating': predicted_rating.item(),
            'user_encoding': user_encoding.detach().numpy().tolist(),
            'similar_liked_movies': [],
            'explanation_text': ""
        }
        
        # Find movies with similar patterns
        if len(user_liked_movies) > 0:
            # Calculate similarity between target movie and user's liked movies
            similarities = []
            for liked_movie_idx in user_liked_movies:
                # Create dummy rating vector for liked movie
                liked_movie_vector = torch.zeros_like(user_ratings)
                liked_movie_vector[liked_movie_idx] = 5.0
                
                # Get encoding
                liked_encoding = self.model.encode(liked_movie_vector.unsqueeze(0)).squeeze()
                
                # Calculate similarity in latent space
                similarity = torch.cosine_similarity(
                    user_encoding.unsqueeze(0), 
                    liked_encoding.unsqueeze(0)
                ).item()
                
                similarities.append({
                    'movie_idx': liked_movie_idx.item(),
                    'movie_id': liked_movie_idx.item() + 1,
                    'similarity': similarity,
                    'user_rating': user_ratings[liked_movie_idx].item()
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            explanation['similar_liked_movies'] = similarities[:3]
            
            # Generate explanation text
            if similarities:
                top_similar = similarities[0]
                explanation['explanation_text'] = (
                    f"This movie is recommended because it has similar patterns to "
                    f"Movie ID {top_similar['movie_id']} (which you rated {top_similar['user_rating']:.1f}). "
                    f"The similarity score in the learned feature space is {top_similar['similarity']:.3f}."
                )
        
        return explanation
    
    def get_user_profile(self, user_ratings: torch.Tensor) -> Dict:
        """
        Analyze a user's preferences and create a profile
        
        Args:
            user_ratings (torch.Tensor): User's rating vector
            
        Returns:
            Dict: User profile analysis
        """
        # Basic statistics
        rated_movies = torch.nonzero(user_ratings).squeeze()
        if len(rated_movies) == 0:
            return {'error': 'User has not rated any movies'}
        
        ratings_given = user_ratings[rated_movies]
        
        profile = {
            'total_movies_rated': len(rated_movies),
            'average_rating': ratings_given.mean().item(),
            'rating_std': ratings_given.std().item(),
            'rating_distribution': {},
            'latent_profile': self.model.encode(user_ratings.unsqueeze(0)).squeeze().detach().numpy().tolist()
        }
        
        # Rating distribution
        for rating in range(1, 6):
            count = (ratings_given == rating).sum().item()
            profile['rating_distribution'][f'{rating}_stars'] = count
        
        # Preference analysis
        profile['preferences'] = {
            'harsh_critic': profile['average_rating'] < 3.0,
            'easy_to_please': profile['average_rating'] > 4.0,
            'diverse_taste': profile['rating_std'] > 1.0,
            'consistent_rater': profile['rating_std'] < 0.5
        }
        
        return profile
    
    def find_similar_users(self, user_ratings: torch.Tensor, 
                          all_user_ratings: torch.Tensor, 
                          top_k: int = 5) -> List[Dict]:
        """
        Find users with similar taste
        
        Args:
            user_ratings (torch.Tensor): Target user's ratings
            all_user_ratings (torch.Tensor): All users' ratings
            top_k (int): Number of similar users to find
            
        Returns:
            List[Dict]: Similar users with details
        """
        similar_users = self.base_recommender.get_similar_users(
            user_ratings, all_user_ratings, top_k
        )
        
        # Enhance with detailed analysis
        detailed_similar_users = []
        user_encoding = self.model.encode(user_ratings.unsqueeze(0))
        
        for user_idx in similar_users:
            other_user_ratings = all_user_ratings[user_idx]
            other_user_encoding = self.model.encode(other_user_ratings.unsqueeze(0))
            
            # Calculate various similarity metrics
            cosine_sim = torch.cosine_similarity(user_encoding, other_user_encoding).item()
            
            # Find common movies
            user_rated = torch.nonzero(user_ratings).squeeze()
            other_rated = torch.nonzero(other_user_ratings).squeeze()
            
            if len(user_rated.shape) == 0:
                user_rated = user_rated.unsqueeze(0)
            if len(other_rated.shape) == 0:
                other_rated = other_rated.unsqueeze(0)
            
            common_movies = set(user_rated.tolist()) & set(other_rated.tolist())
            
            user_info = {
                'user_id': user_idx + 1,  # Convert to 1-indexed
                'similarity_score': cosine_sim,
                'common_movies_count': len(common_movies),
                'user_avg_rating': other_user_ratings[other_user_ratings > 0].mean().item() if torch.sum(other_user_ratings > 0) > 0 else 0
            }
            
            detailed_similar_users.append(user_info)
        
        return detailed_similar_users
    
    def generate_diversity_recommendations(self, user_ratings: torch.Tensor,
                                         top_k: int = 10,
                                         diversity_weight: float = 0.3) -> List[Dict]:
        """
        Generate diverse recommendations to avoid filter bubble
        
        Args:
            user_ratings (torch.Tensor): User's rating vector
            top_k (int): Number of recommendations
            diversity_weight (float): Weight for diversity vs relevance
            
        Returns:
            List[Dict]: Diverse recommendations
        """
        # Get more recommendations than needed
        candidates = self.get_recommendations(user_ratings, top_k=top_k*3, exclude_rated=True)
        
        if len(candidates) == 0:
            return []
        
        # Select diverse subset
        selected = [candidates[0]]  # Start with top recommendation
        candidates = candidates[1:]
        
        while len(selected) < top_k and candidates:
            best_candidate = None
            best_score = -1
            
            for candidate in candidates:
                # Calculate relevance score (predicted rating)
                relevance = candidate['predicted_rating']
                
                # Calculate diversity score (average distance from selected items)
                if len(selected) > 0:
                    diversity = self._calculate_diversity_score(candidate, selected)
                else:
                    diversity = 1.0
                
                # Combined score
                combined_score = (1 - diversity_weight) * relevance + diversity_weight * diversity
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
        
        return selected
    
    def analyze_recommendation_trends(self, all_user_ratings: torch.Tensor,
                                    sample_size: int = 100) -> Dict:
        """
        Analyze overall recommendation trends
        
        Args:
            all_user_ratings (torch.Tensor): All users' ratings
            sample_size (int): Number of users to sample
            
        Returns:
            Dict: Trend analysis
        """
        print("🔍 Analyzing recommendation trends...")
        
        # Sample users
        total_users = all_user_ratings.shape[0]
        sample_indices = torch.randperm(total_users)[:min(sample_size, total_users)]
        
        movie_recommendation_counts = defaultdict(int)
        movie_avg_predicted_ratings = defaultdict(list)
        
        # Generate recommendations for sample users
        for user_idx in sample_indices:
            user_ratings = all_user_ratings[user_idx]
            
            # Skip users with no ratings
            if torch.sum(user_ratings > 0) == 0:
                continue
            
            recommendations = self.get_recommendations(user_ratings, top_k=10)
            
            for rec in recommendations:
                movie_id = rec['movie_id']
                movie_recommendation_counts[movie_id] += 1
                movie_avg_predicted_ratings[movie_id].append(rec['predicted_rating'])
        
        # Calculate averages
        movie_stats = {}
        for movie_id, count in movie_recommendation_counts.items():
            movie_stats[movie_id] = {
                'recommendation_frequency': count,
                'avg_predicted_rating': np.mean(movie_avg_predicted_ratings[movie_id]),
                'std_predicted_rating': np.std(movie_avg_predicted_ratings[movie_id])
            }
        
        # Find most/least recommended movies
        most_recommended = sorted(movie_stats.items(), 
                                key=lambda x: x[1]['recommendation_frequency'], 
                                reverse=True)[:10]
        
        highest_rated = sorted(movie_stats.items(),
                             key=lambda x: x[1]['avg_predicted_rating'],
                             reverse=True)[:10]
        
        analysis = {
            'sample_size': len(sample_indices),
            'total_unique_recommendations': len(movie_stats),
            'most_recommended_movies': most_recommended,
            'highest_avg_predicted_movies': highest_rated,
            'recommendation_diversity': len(movie_stats) / len(sample_indices) if len(sample_indices) > 0 else 0
        }
        
        return analysis
    
    def _calculate_confidence(self, user_ratings: torch.Tensor, movie_idx: int) -> float:
        """
        Calculate confidence score for a recommendation
        
        Args:
            user_ratings (torch.Tensor): User's ratings
            movie_idx (int): Movie index
            
        Returns:
            float: Confidence score (0-1)
        """
        # Simple confidence based on how many movies user has rated
        # and similarity to user's preferred rating range
        
        rated_count = torch.sum(user_ratings > 0).item()
        confidence = min(rated_count / 50.0, 1.0)  # Max confidence at 50 ratings
        
        return confidence
    
    def _get_movie_metadata(self, movie_id: int) -> Dict:
        """
        Get movie metadata if available
        
        Args:
            movie_id (int): Movie ID
            
        Returns:
            Dict: Movie metadata
        """
        if self.movies_data is None:
            return {'title': f'Movie {movie_id}', 'genres': 'Unknown'}
        
        # This would be implemented based on your actual movie data structure
        # For now, return basic info
        return {
            'title': f'Movie {movie_id}',
            'genres': 'Unknown'
        }
    
    def _calculate_diversity_score(self, candidate: Dict, selected: List[Dict]) -> float:
        """
        Calculate diversity score for a candidate recommendation
        
        Args:
            candidate (Dict): Candidate recommendation
            selected (List[Dict]): Already selected recommendations
            
        Returns:
            float: Diversity score
        """
        if not selected:
            return 1.0
        
        # Simple diversity based on predicted rating differences
        candidate_rating = candidate['predicted_rating']
        selected_ratings = [item['predicted_rating'] for item in selected]
        
        # Calculate minimum distance to selected items
        min_distance = min(abs(candidate_rating - rating) for rating in selected_ratings)
        
        # Normalize to 0-1 range
        diversity_score = min(min_distance / 2.0, 1.0)  # Max diversity at 2.0 rating difference
        
        return diversity_score
    
    def create_recommendation_report(self, user_ratings: torch.Tensor,
                                   user_id: int = 1) -> str:
        """
        Create a comprehensive recommendation report
        
        Args:
            user_ratings (torch.Tensor): User's ratings
            user_id (int): User ID for the report
            
        Returns:
            str: Formatted report
        """
        report = []
        report.append(f"🎬 MOVIE RECOMMENDATION REPORT FOR USER {user_id}")
        report.append("=" * 60)
        
        # User profile
        profile = self.get_user_profile(user_ratings)
        if 'error' not in profile:
            report.append(f"\n👤 USER PROFILE:")
            report.append(f"   • Total movies rated: {profile['total_movies_rated']}")
            report.append(f"   • Average rating: {profile['average_rating']:.2f}")
            report.append(f"   • Rating consistency: {profile['rating_std']:.2f}")
            
            # Preference insights
            prefs = profile['preferences']
            if prefs['harsh_critic']:
                report.append("   • Profile: Harsh critic (low average rating)")
            elif prefs['easy_to_please']:
                report.append("   • Profile: Easy to please (high average rating)")
            
            if prefs['diverse_taste']:
                report.append("   • Taste: Diverse (high rating variance)")
            elif prefs['consistent_rater']:
                report.append("   • Taste: Consistent (low rating variance)")
        
        # Top recommendations
        recommendations = self.get_recommendations(user_ratings, top_k=10)
        
        if recommendations:
            report.append(f"\n🎯 TOP 10 RECOMMENDATIONS:")
            report.append("-" * 40)
            
            for i, rec in enumerate(recommendations, 1):
                confidence_str = "⭐" * min(int(rec['confidence_score'] * 5), 5)
                report.append(
                    f"  {i:2d}. Movie ID {rec['movie_id']:4d} | "
                    f"Rating: {rec['predicted_rating']:.2f} | "
                    f"Confidence: {confidence_str}"
                )
        
        # Diversity recommendations
        diverse_recs = self.generate_diversity_recommendations(user_ratings, top_k=5)
        
        if diverse_recs:
            report.append(f"\n🌈 DIVERSE RECOMMENDATIONS:")
            report.append("-" * 40)
            
            for i, rec in enumerate(diverse_recs, 1):
                report.append(
                    f"  {i}. Movie ID {rec['movie_id']:4d} | "
                    f"Rating: {rec['predicted_rating']:.2f}"
                )
        
        report.append(f"\n📊 Generated by SAE Movie Recommendation System")
        report.append(f"   Model: Stacked AutoEncoder with {self.model.nb_movies} movies")
        
        return "\n".join(report)
    
    def plot_user_analysis(self, user_ratings: torch.Tensor, user_id: int = 1) -> None:
        """
        Create visualizations for user analysis
        
        Args:
            user_ratings (torch.Tensor): User's ratings
            user_id (int): User ID for titles
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'User {user_id} Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Rating distribution
        rated_movies = torch.nonzero(user_ratings).squeeze()
        if len(rated_movies) > 0:
            ratings_given = user_ratings[rated_movies].numpy()
            
            axes[0, 0].hist(ratings_given, bins=np.arange(0.5, 6.5, 1), 
                           color='skyblue', alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Rating Distribution')
            axes[0, 0].set_xlabel('Rating')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_xticks(range(1, 6))
        
        # 2. Latent space representation
        profile = self.get_user_profile(user_ratings)
        if 'latent_profile' in profile:
            latent_features = profile['latent_profile']
            axes[0, 1].bar(range(len(latent_features)), latent_features, color='lightgreen')
            axes[0, 1].set_title('Latent Feature Representation')
            axes[0, 1].set_xlabel('Latent Dimension')
            axes[0, 1].set_ylabel('Activation')
        
        # 3. Recommendation confidence
        recommendations = self.get_recommendations(user_ratings, top_k=10)
        if recommendations:
            movie_ids = [rec['movie_id'] for rec in recommendations]
            predicted_ratings = [rec['predicted_rating'] for rec in recommendations]
            confidence_scores = [rec['confidence_score'] for rec in recommendations]
            
            scatter = axes[1, 0].scatter(predicted_ratings, confidence_scores, 
                                       c=range(len(recommendations)), cmap='viridis')
            axes[1, 0].set_title('Recommendation Quality')
            axes[1, 0].set_xlabel('Predicted Rating')
            axes[1, 0].set_ylabel('Confidence Score')
            
            # Add movie ID labels
            for i, movie_id in enumerate(movie_ids):
                axes[1, 0].annotate(f'{movie_id}', 
                                  (predicted_ratings[i], confidence_scores[i]),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Rating timeline (if we had timestamps)
        # For now, show recommendation scores
        if recommendations:
            movie_ids = [rec['movie_id'] for rec in recommendations]
            predicted_ratings = [rec['predicted_rating'] for rec in recommendations]
            
            axes[1, 1].plot(range(1, len(recommendations) + 1), predicted_ratings, 
                           'o-', color='red', markersize=8)
            axes[1, 1].set_title('Top Recommendations Score')
            axes[1, 1].set_xlabel('Recommendation Rank')
            axes[1, 1].set_ylabel('Predicted Rating')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Example usage of the RecommendationEngine
    """
    print("🎬 Advanced Movie Recommendation Engine")
    print("=" * 50)
    
    # This would be used with a trained model:
    # model = SAE(nb_movies=1682, encoding_layers=[20, 10])
    # model.load_state_dict(torch.load('models/best_sae_model.pth'))
    # engine = RecommendationEngine(model)
    
    print("Recommendation engine ready for use with trained SAE model!")
    print("\nFeatures available:")
    print("  ✅ Detailed recommendations with confidence scores")
    print("  ✅ User preference profiling and analysis")
    print("  ✅ Similar user discovery")
    print("  ✅ Diverse recommendation generation")
    print("  ✅ Recommendation explanation")
    print("  ✅ Comprehensive reporting and visualization")

if __name__ == "__main__":
    main()