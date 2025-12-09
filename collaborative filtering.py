import numpy as np
import pandas as pd

# Fix random seed for reproducibility
np.random.seed(42)

# ----- Step A1: create movies list -----
movies = [
    "The Matrix (Action)",
    "Inception (Sci-Fi)",
    "The Godfather (Crime)",
    "Toy Story (Animation)",
    "Titanic (Romance)",
    "The Dark Knight (Action)",
    "Forrest Gump (Drama)",
    "Pulp Fiction (Crime)",
    "The Shawshank Redemption (Drama)",
    "Interstellar (Sci-Fi)",
    "Gladiator (Action)",
    "Finding Nemo (Animation)",
    "The Lion King (Animation)",
    "Avengers: Endgame (Action)",
    "La La Land (Musical)",
    "The Social Network (Drama)",
    "Parasite (Thriller)",
    "Spirited Away (Animation)",
    "Joker (Crime)",
    "Mad Max: Fury Road (Action)"
]

num_users = 15
num_movies = len(movies)

# Create empty ratings matrix (users x movies), NaN = no rating yet
user_ids = [f"U{i+1}" for i in range(num_users)]
movie_ids = [f"M{j+1}" for j in range(num_movies)]
ratings = pd.DataFrame(np.nan, index=user_ids, columns=movie_ids)


rng = np.random.default_rng(42)

def assign_ratings_to_group(user_range, min_movies, max_movies):
    """
    Assign random ratings (1-5) to each user in the given range.
    user_range is a range of row indices in the ratings DataFrame.
    """
    for u in user_range:
        # how many movies this user will rate
        k = rng.integers(min_movies, max_movies + 1)
        # choose k distinct movies
        movie_indices = rng.choice(num_movies, size=k, replace=False)
        for m in movie_indices:
            ratings.iloc[u, m] = rng.integers(1, 6)  # rating between 1 and 5

# A.2 – Assign ratings
# Users 1–5: between 8–10 movies
assign_ratings_to_group(range(0, 5), 8, 10)

# Users 6–10: between 4–6 movies
assign_ratings_to_group(range(5, 10), 4, 6)

# Users 11–15 (new users): between 2–3 movies
assign_ratings_to_group(range(10, 15), 2, 3)

print("=== Raw ratings matrix (users x movies, 1-5, NaN = no rating) ===")
print(ratings)

# ----- Step A3: normalization with centering -----
# User mean rating (ignoring NaNs)
user_means = ratings.mean(axis=1, skipna=True)

# Subtract each user's mean from their ratings
ratings_norm = ratings.sub(user_means, axis=0)

print("\n=== User mean ratings ===")
print(user_means)

print("\n=== Normalized (centered) ratings matrix ===")
print(ratings_norm)

# ----- Step B: user-user similarity for first 10 users -----
def cosine_similarity_on_overlap(vec1, vec2):
    """
    Cosine similarity computed only on positions where both vectors are not NaN.
    vec1, vec2 are 1D numpy arrays.
    """
    mask = (~np.isnan(vec1)) & (~np.isnan(vec2))
    if mask.sum() < 2:
        # not enough overlapping movies to compute a reliable similarity
        return 0.0
    v1 = vec1[mask]
    v2 = vec2[mask]
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

similarities = []
first_10_users = user_ids[:10]

for i in range(len(first_10_users)):
    for j in range(i + 1, len(first_10_users)):
        u_i = first_10_users[i]
        u_j = first_10_users[j]
        sim = cosine_similarity_on_overlap(
            ratings_norm.loc[u_i].values,
            ratings_norm.loc[u_j].values
        )
        similarities.append((u_i, u_j, sim))

# Sort by similarity descending
similarities.sort(key=lambda x: x[2], reverse=True)

print("\n=== Top 3 most similar user pairs among the first 10 users ===")
for u_i, u_j, sim in similarities[:3]:
    print(f"{u_i} - {u_j}: similarity = {sim:.3f}")

# ----- Step C: recommendations for new users (U11–U15) -----
def recommend_for_new_user(new_user_id, k_neighbors=3, top_n_movies=3):
    """
    Recommend movies for a new user using user-based collaborative filtering.
    We use the k most similar users (from the first 10 users) and compute
    weighted predicted ratings for unseen movies.
    """
    # Similarities from new user to each of the first 10 users
    sims_to_others = []
    for u in first_10_users:
        sim = cosine_similarity_on_overlap(
            ratings_norm.loc[new_user_id].values,
            ratings_norm.loc[u].values
        )
        sims_to_others.append((u, sim))

    # sort by similarity desc, take top k
    sims_to_others.sort(key=lambda x: x[1], reverse=True)
    neighbors = [item for item in sims_to_others[:k_neighbors] if item[1] > 0]

    print(f"\n--- Recommendations for {new_user_id} ---")
    if not neighbors:
        print("Not enough similar users to make a recommendation.")
        return

    print("Most similar existing users:")
    for u, s in neighbors:
        print(f"  {u} (similarity = {s:.3f})")

    # Movies that the new user did NOT rate
    new_user_ratings = ratings.loc[new_user_id]
    candidate_movies = [m for m in ratings.columns if pd.isna(new_user_ratings[m])]

    user_mean = user_means.loc[new_user_id]
    movie_scores = []

    for m in candidate_movies:
        num = 0.0
        denom = 0.0
        for neighbor_id, sim in neighbors:
            r_neighbor_norm = ratings_norm.loc[neighbor_id, m]
            if not np.isnan(r_neighbor_norm):
                num += sim * r_neighbor_norm
                denom += abs(sim)
        if denom > 0:
            pred_norm = num / denom            # predicted normalized rating
            pred_rating = user_mean + pred_norm  # convert back to absolute scale
            movie_scores.append((m, pred_rating))

    if not movie_scores:
        print("Not enough data to predict ratings for unseen movies.")
        return

    # sort by predicted rating desc
    movie_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n_movies} recommended movies for {new_user_id}:")
    for m, score in movie_scores[:top_n_movies]:
        idx = int(m[1:]) - 1  # convert "M5" -> 4
        print(f"  {m} - {movies[idx]} (predicted rating = {score:.2f})")

# Run recommendations for users U11–U15
for uid in user_ids[10:]:
    recommend_for_new_user(uid)
