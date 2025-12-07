#!/usr/bin/env python3
"""
Split viewer_interactions.db into three datasets:
- Training: 70%
- Validation: 15%
- Testing: 15%

Uses fully random division.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility (optional - remove if you want different splits each time)
np.random.seed(42)

# Paths
SOURCE_DB = "./viewer_interactions.db"
TRAIN_DB = "./viewer_interactions_train.db"
VAL_DB = "./viewer_interactions_val.db"
TEST_DB = "./viewer_interactions_test.db"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1.0"

print("=" * 60)
print("Database Split Script")
print("=" * 60)

# Connect to source database
print(f"\n1. Loading data from {SOURCE_DB}...")
conn_source = sqlite3.connect(SOURCE_DB)

# Load main data table
viewer_ratings = pd.read_sql("SELECT * FROM viewer_ratings", conn_source)
print(f"   Loaded {len(viewer_ratings):,} rows from viewer_ratings")

# Load reference tables (these will be copied to all splits)
movies = pd.read_sql("SELECT * FROM movies", conn_source)
data_dictionary = pd.read_sql("SELECT * FROM data_dictionary", conn_source)
print(f"   Loaded {len(movies):,} rows from movies")
print(f"   Loaded {len(data_dictionary):,} rows from data_dictionary")

conn_source.close()

# Random shuffle
print(f"\n2. Randomly shuffling data...")
viewer_ratings = viewer_ratings.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Calculate split indices
total_rows = len(viewer_ratings)
train_end = int(total_rows * TRAIN_RATIO)
val_end = train_end + int(total_rows * VAL_RATIO)

print(f"   Total rows: {total_rows:,}")
print(f"   Training: {train_end:,} rows ({TRAIN_RATIO*100:.1f}%)")
print(f"   Validation: {val_end - train_end:,} rows ({VAL_RATIO*100:.1f}%)")
print(f"   Testing: {total_rows - val_end:,} rows ({TEST_RATIO*100:.1f}%)")

# Split the data
train_ratings = viewer_ratings.iloc[:train_end].copy()
val_ratings = viewer_ratings.iloc[train_end:val_end].copy()
test_ratings = viewer_ratings.iloc[val_end:].copy()

# Function to create a database with data
def create_database(db_path, ratings_df, split_name):
    """Create a database file with the given ratings and reference tables."""
    print(f"\n3. Creating {split_name} database: {db_path}")
    
    # Remove existing file if it exists
    if Path(db_path).exists():
        Path(db_path).unlink()
        print(f"   Removed existing {db_path}")
    
    conn = sqlite3.connect(db_path)
    
    # Write ratings
    ratings_df.to_sql("viewer_ratings", conn, index=False, if_exists="replace")
    print(f"   Written {len(ratings_df):,} rows to viewer_ratings")
    
    # Write reference tables
    movies.to_sql("movies", conn, index=False, if_exists="replace")
    data_dictionary.to_sql("data_dictionary", conn, index=False, if_exists="replace")
    print(f"   Written reference tables (movies, data_dictionary)")
    
    # Calculate and write user_statistics
    print(f"   Calculating user_statistics...")
    user_stats = ratings_df.groupby("customer_id").agg({
        "rating": ["count", "mean", "std", "min", "max"],
        "movie_id": "nunique",
        "date": ["min", "max"]
    }).reset_index()
    
    user_stats.columns = [
        "customer_id",
        "total_ratings",
        "avg_rating",
        "std_rating",
        "min_rating",
        "max_rating",
        "unique_movies",
        "first_rating_date",
        "last_rating_date"
    ]
    
    # Calculate activity_days
    user_stats["first_rating_date"] = pd.to_datetime(user_stats["first_rating_date"])
    user_stats["last_rating_date"] = pd.to_datetime(user_stats["last_rating_date"])
    user_stats["activity_days"] = (user_stats["last_rating_date"] - user_stats["first_rating_date"]).dt.days
    user_stats["activity_days"] = user_stats["activity_days"].fillna(0)
    
    # Convert dates back to strings
    user_stats["first_rating_date"] = user_stats["first_rating_date"].dt.strftime("%Y-%m-%d")
    user_stats["last_rating_date"] = user_stats["last_rating_date"].dt.strftime("%Y-%m-%d")
    
    user_stats.to_sql("user_statistics", conn, index=False, if_exists="replace")
    print(f"   Written {len(user_stats):,} rows to user_statistics")
    
    # Calculate and write movie_statistics
    print(f"   Calculating movie_statistics...")
    movie_stats = ratings_df.groupby("movie_id").agg({
        "rating": ["count", "mean", "std", "min", "max"],
        "customer_id": "nunique",
        "date": ["min", "max"]
    }).reset_index()
    
    movie_stats.columns = [
        "movie_id",
        "total_ratings",
        "avg_rating",
        "std_rating",
        "min_rating",
        "max_rating",
        "unique_users",
        "first_rating_date",
        "last_rating_date"
    ]
    
    # Merge with movies to get year_of_release and title
    movie_stats = movie_stats.merge(
        movies[["movie_id", "year_of_release", "title"]],
        on="movie_id",
        how="left"
    )
    
    # Convert dates
    movie_stats["first_rating_date"] = pd.to_datetime(movie_stats["first_rating_date"]).dt.strftime("%Y-%m-%d")
    movie_stats["last_rating_date"] = pd.to_datetime(movie_stats["last_rating_date"]).dt.strftime("%Y-%m-%d")
    
    movie_stats.to_sql("movie_statistics", conn, index=False, if_exists="replace")
    print(f"   Written {len(movie_stats):,} rows to movie_statistics")
    
    conn.close()
    print(f"   ✓ {split_name} database created successfully")

# Create all three databases
create_database(TRAIN_DB, train_ratings, "Training")
create_database(VAL_DB, val_ratings, "Validation")
create_database(TEST_DB, test_ratings, "Testing")

print("\n" + "=" * 60)
print("Split Complete!")
print("=" * 60)
print(f"\nCreated files:")
print(f"  - {TRAIN_DB} ({len(train_ratings):,} ratings)")
print(f"  - {VAL_DB} ({len(val_ratings):,} ratings)")
print(f"  - {TEST_DB} ({len(test_ratings):,} ratings)")
print(f"\nTotal: {len(train_ratings) + len(val_ratings) + len(test_ratings):,} ratings")
print(f"Original: {total_rows:,} ratings")
print(f"\n✓ All databases created successfully!")

