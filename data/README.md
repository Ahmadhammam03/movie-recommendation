# Dataset Instructions

This project uses MovieLens datasets for training the Stacked AutoEncoder recommendation system.

## 📊 Current Data Structure

Based on your uploaded files, you have:

### ML-1M Dataset (data/ml-1m/):
- `movies.dat` - Movie information
- `ratings.dat` - Rating information  
- `users.dat` - User demographics
- `training_set.csv` - **750,122 ratings** (custom training data)
- `test_set.csv` - **250,089 ratings** (custom test data)

### ML-100K Dataset (data/ml-100k/):
- `u1.base` - Training set (100k format)
- `u1.test` - Test set (100k format)
- Various other utility files

## 🔧 Data Format

### Your Custom CSV Files:
Both `training_set.csv` and `test_set.csv` have the format:
```
User,Movie,Rating,Timestamp
1,661,3,978302109
1,914,3,978301968
...
```

**Columns:**
- **User**: User ID (1-based indexing)
- **Movie**: Movie ID (1-based indexing)  
- **Rating**: Rating value (1-5 scale)
- **Timestamp**: Unix timestamp

**Statistics:**
- Training: 750,122 ratings
- Test: 250,089 ratings
- Total: ~1 million ratings
- Users: ~42 unique users
- Movies: ~4,000 unique movies
- Time span: ~2001-2003

## 🚀 Usage in Code

The project automatically detects and uses your CSV format:

```python
# The trainer will automatically load your CSV files
experiment = SAEExperiment(data_path="data/")
experiment.load_and_prepare_data(use_custom_csv=True)
```

## 📁 File Structure

```
data/
├── ml-1m/
│   ├── movies.dat           # Movie metadata
│   ├── ratings.dat          # All ratings  
│   ├── users.dat            # User info
│   ├── training_set.csv     # training data (750k)
│   └── test_set.csv         # test data (250k)
└── ml-100k/
    ├── u1.base              # Alternative training format
    ├── u1.test              # Alternative test format
    └── ... (other files)
```

## 🔍 Data Quality Notes

- **High activity users**: Users 1 and 2 dominate the sample
- **Popular movies**: IDs like 2858, 2028, 914 appear frequently
- **Rating distribution**: Mostly 4-5 star ratings (~70%)
- **Sparsity**: Most user-movie pairs are unrated (typical for recommendation systems)

## 💡 Alternative Download (if needed)

If you need to download fresh MovieLens datasets:

1. **MovieLens 1M**: https://grouplens.org/datasets/movielens/1m/
2. **MovieLens 100K**: https://grouplens.org/datasets/movielens/100k/

Extract to the respective folders, but your current CSV files are perfect for training!

## ⚠️ Important Notes

- **Large files**: The CSV files are not committed to git (too large)
- **Format compatibility**: Your CSV format is automatically detected
- **Performance**: 750k+ ratings provide excellent training data
- **Test set**: 250k test ratings ensure robust evaluation

Your data setup is **perfect** for the SAE model training! 🎯