# NYC Restaurant Recommender

A multi-page Streamlit app for discovering New York City restaurants with semantic search, personalized ranking, Google Places enrichment, and cluster-based exploration.

## Overview

This project turns raw NYC restaurant inspection data into a richer restaurant discovery experience.

It combines:

- NYC Department of Health inspection data,
- Google Places ratings, summaries, photos, and Maps links,
- sentence-transformer embeddings for semantic search,
- profile-aware recommendation scoring,
- clustering and 3D visualization for exploration.

The result is an app where a user can describe what they want in plain English, save likes and preferences, and browse restaurants through search, recommendations, a GIS-style map, and a PCA cluster explorer.

## What The App Does

The main user-facing experience lives in [`Home.py`](/Users/rahuladusumalli/Desktop/ML%20Project/Home.py) and the pages inside [`pages/`](/Users/rahuladusumalli/Desktop/ML%20Project/pages).

### Home

[`Home.py`](/Users/rahuladusumalli/Desktop/ML%20Project/Home.py) is the main landing page. It:

- loads the prepared restaurant search dataset,
- initializes the user profile sidebar,
- lets the user run a natural-language search immediately,
- exposes cluster filtering when clustering has already been computed.

### Semantic Search

[`pages/4_🔎_Semantic_Search.py`](/Users/rahuladusumalli/Desktop/ML%20Project/pages/4_%F0%9F%94%8E_Semantic_Search.py) is the dedicated search page. It:

- accepts natural-language restaurant queries,
- ranks results using semantic, lexical, quality, and profile signals,
- shows Google ratings, review counts, price levels, and photos,
- lets the user save restaurants to their profile.

### Recommendations

[`pages/3_🔮_Recommendations.py`](/Users/rahuladusumalli/Desktop/ML%20Project/pages/3_%F0%9F%94%AE_Recommendations.py) provides cluster-based recommendations. It:

- reclusters the runtime dataset,
- predicts the user’s taste cluster from saved history,
- ranks restaurants by user affinity within that cluster,
- summarizes the cluster profile and top picks.

### GIS Map

[`pages/1_📍_GIS_Map.py`](/Users/rahuladusumalli/Desktop/ML%20Project/pages/1_%F0%9F%93%8D_GIS_Map.py) plots restaurants geographically. It:

- colors restaurants by cluster,
- supports scatter and 3D column views,
- can highlight the user’s predicted cluster,
- can send a selected cluster back to the search pages.

### PCA Explorer

[`pages/2_📊_PCA_Explorer.py`](/Users/rahuladusumalli/Desktop/ML%20Project/pages/2_%F0%9F%93%8A_PCA_Explorer.py) visualizes the clustered dataset in 3D feature space using PCA.

## Data Sources

### 1. NYC Open Data

The base dataset comes from the NYC restaurant inspection endpoint:

- `https://data.cityofnewyork.us/resource/43nn-pn8j.json`

The app uses fields such as:

- restaurant ID (`camis`)
- restaurant name (`dba`)
- borough
- address
- cuisine description
- health grade
- inspection score
- latitude and longitude

### 2. Google Places API

Restaurants are enriched through Google Places with:

- Google rating
- review count
- price level
- editorial summary
- Maps URL
- photo reference

## Models And ML Components

## Semantic Embedding Model

The primary embedding model in the modular search pipeline is loaded in [`utils/search.py`](/Users/rahuladusumalli/Desktop/ML%20Project/utils/search.py):

- `sentence-transformers/all-mpnet-base-v2`

That model produces `768`-dimensional embeddings. The cached prepared embedding matrix in this repo confirms that:

- `data/cache/embeddings_prepared_750_672.npy` has shape `(672, 768)`

## Legacy Embedding Model

The older compatibility app in [`app.py`](/Users/rahuladusumalli/Desktop/ML%20Project/app.py) still uses:

- `sentence-transformers/all-MiniLM-L6-v2`

That path is still available, but the main maintained flow is the multi-page `Home.py` app.

## Search Ranking

The main ranking function in [`utils/search.py`](/Users/rahuladusumalli/Desktop/ML%20Project/utils/search.py) combines:

- semantic similarity,
- lexical keyword overlap,
- quality scoring,
- profile scoring.

The blended score is:

- `0.50 * semantic`
- `0.20 * lexical`
- `0.15 * quality`
- `0.15 * profile`

## Profile Scoring

User preference scoring is implemented in [`utils/user_profile.py`](/Users/rahuladusumalli/Desktop/ML%20Project/utils/user_profile.py).

It uses:

- favorite cuisines,
- preferred boroughs,
- budget,
- minimum acceptable health grade,
- spice tolerance,
- adventurousness,
- saved likes.

## Clustering

The clustering pipeline in [`utils/clustering.py`](/Users/rahuladusumalli/Desktop/ML%20Project/utils/clustering.py) uses:

- `StandardScaler`
- `KMeans`
- `MiniBatchKMeans`
- `PCA`
- optional `UMAP` when available

Features include:

- cuisine one-hot encoding,
- normalized price tier,
- normalized average rating,
- log-scaled review count,
- normalized latitude and longitude,
- optional tag features,
- user-affinity augmentation.

## How The Data Pipeline Works

The current runtime flow is:

1. Load a prepared cached restaurant dataset if available.
2. If needed, fetch NYC base data.
3. Enrich a sample of restaurants with Google Places data.
4. Build a natural-language description for each restaurant.
5. Load cached embeddings or compute them.
6. Convert the prepared search dataframe into a runtime dataframe with the extra columns the clustering pages expect.
7. Reuse that shared runtime dataframe across search, clustering, map, PCA, and recommendations.

That shared conversion happens in [`utils/search_assets.py`](/Users/rahuladusumalli/Desktop/ML%20Project/utils/search_assets.py).

## Caching

Prepared artifacts live in [`data/cache/`](/Users/rahuladusumalli/Desktop/ML%20Project/data/cache).

Examples currently in this repo:

- `prepared_search_750.pkl`
- `enriched_restaurants_750.pkl`
- `embeddings_prepared_750_672.npy`
- `embeddings_semantic_750_670.npy`

In the current workspace:

- the prepared search dataframe has shape `(672, 23)`
- the runtime dataframe built from it has shape `(672, 31)`
- the main prepared embedding matrix has shape `(672, 768)`

## Profiles And Persistence

User profiles are stored in:

- [`data/user_profiles.json`](/Users/rahuladusumalli/Desktop/ML%20Project/data/user_profiles.json)

The profile system now supports:

- selecting an existing profile,
- creating a new profile,
- saving cuisine, borough, budget, grade, vibe, spice, and adventurousness preferences,
- saving liked restaurants from the UI,
- translating saved likes into cluster history for the recommendation pages.

## How To Run

## 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure your Google Places key

Create:

- `.streamlit/secrets.toml`

With:

```toml
GOOGLE_API_KEY = "your_google_places_api_key"
```

The code still has a fallback key path for compatibility, but a project-specific key in Streamlit secrets is the recommended setup.

## 4. Start the main app

```bash
streamlit run Home.py
```

You can also launch the older single-page compatibility version with:

```bash
streamlit run app.py
```

## What Was Fixed

The current codebase now has a consistent runtime flow across the main app:

- the missing profile helper functions were restored in [`utils/user_profile.py`](/Users/rahuladusumalli/Desktop/ML%20Project/utils/user_profile.py),
- the circular import issue in [`utils/clustering.py`](/Users/rahuladusumalli/Desktop/ML%20Project/utils/clustering.py) was removed,
- the newer pages and the clustering pages now share one runtime dataframe shape,
- cluster selection from the map can flow back into the search pages,
- pages can auto-load prepared data instead of requiring a hidden “visit Home first” step,
- the prepared cache is now used before attempting a fresh NYC API fetch,
- semantic search now degrades gracefully when the transformer model is not locally available.

## Offline / Fallback Behavior

The app is designed to behave sensibly when some resources are already cached but the embedding model is not locally available yet.

If the search model cannot be loaded:

- cached restaurant embeddings are still reused when available,
- query-time semantic scoring falls back cleanly,
- lexical, quality, and profile scoring still produce usable rankings.

This makes the app much more resilient in restricted or partially offline environments.

## Main Dependencies

From [`requirements.txt`](/Users/rahuladusumalli/Desktop/ML%20Project/requirements.txt):

- `streamlit`
- `pandas`
- `numpy`
- `requests`
- `sentence-transformers`
- `scikit-learn`
- `torch`
- `transformers`
- `pydeck`
- `plotly`
- `joblib`
- `pyarrow`
- `python-dotenv`

## Project Structure

```text
ML Project/
├── Home.py
├── app.py
├── requirements.txt
├── data/
│   ├── cache/
│   └── user_profiles.json
├── pages/
│   ├── 1_📍_GIS_Map.py
│   ├── 2_📊_PCA_Explorer.py
│   ├── 3_🔮_Recommendations.py
│   └── 4_🔎_Semantic_Search.py
└── utils/
    ├── clustering.py
    ├── data.py
    ├── google_places.py
    ├── recommendation_engine.py
    ├── search.py
    ├── search_assets.py
    └── user_profile.py
```

## Summary

This project is a full restaurant discovery prototype for NYC that uses real public data, external API enrichment, embeddings, personalized scoring, and clustering-based visualization in one Streamlit app.

The main maintained app entrypoint is [`Home.py`](/Users/rahuladusumalli/Desktop/ML%20Project/Home.py), and the codebase now shares a consistent data flow between search, recommendations, clustering, and saved user preferences.
