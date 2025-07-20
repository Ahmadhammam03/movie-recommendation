# In your notebook or as a script
from src.trainer import SAEExperiment

# Initialize experiment  
experiment = SAEExperiment(data_path="data/", model_save_path="models/")

# Run complete pipeline
experiment.load_and_prepare_data(use_custom_csv=True)  # Uses YOUR CSV files
experiment.initialize_model(encoding_layers=[20, 10])
experiment.train_model(nb_epochs=200)
experiment.evaluate_model()
experiment.demonstrate_recommendations()
experiment.save_model("best_sae_model.pth")