import os
import time
import streamlit as st
GOOGLE_API_KEY = "AIzaSyBM_Td0_NgsHAmjOldP7AVH5pySmZH--I8"

import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.user_profile import init_session_state

st.set_page_config(
    page_title="NYC Restaurant Semantic Search",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

NYC_DOHMH_API  = "https://data.cityofnewyork.us/resource/43nn-pn8j.json"
PLACES_SEARCH  = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACES_DETAILS = "https://maps.googleapis.com/maps/api/place/details/json"
PLACES_PHOTO   = "https://maps.googleapis.com/maps/api/place/photo"
ENRICHED_CACHE_PATH = "data/enriched_restaurants.parquet"

CUISINE_KEYWORDS = {
    "American":       "burgers steaks BBQ comfort food fries classic American diner",
    "Italian":        "pasta pizza risotto tiramisu antipasto gelato wine",
    "Chinese":        "dim sum dumplings noodles stir fry Peking duck Cantonese Sichuan",
    "Japanese":       "sushi ramen sashimi tempura udon miso izakaya omakase",
    "Mexican":        "tacos burritos enchiladas guacamole margaritas salsa tamales",
    "French":         "croissants baguette coq au vin escargot crepes bistro wine",
    "Indian":         "curry tandoori naan biryani masala daal spices South Asian",
    "Thai":           "pad thai green curry satay lemongrass coconut milk noodles",
    "Korean":         "bibimbap kimchi bulgogi Korean BBQ tofu banchan galbi",
    "Vietnamese":     "pho banh mi spring rolls lemongrass rice noodle soup bun",
    "Mediterranean":  "hummus falafel gyros shawarma olive oil mezze halloumi",
    "Latin":          "empanadas rice beans plantains ceviche Latin fusion churros",
    "Cafe":           "coffee espresso brunch pastries sandwiches light fare",
    "Pizza":          "New York slice thin crust mozzarella tomato sauce calzone",
    "Seafood":        "lobster oysters shrimp clams crab fresh fish grilled seafood",
    "Bakery":         "bread pastries croissants cakes artisan baked goods sourdough",
    "Steak":          "steakhouse prime rib tenderloin ribeye porterhouse dry-aged beef",
    "Vegetarian":     "vegan plant-based tofu salads grain bowls organic healthy",
    "Middle Eastern": "kebab hummus falafel tahini shawarma pita manakeesh",
    "Spanish":        "tapas paella sangria chorizo jamon iberico pintxos",
    "Juice Bar":      "smoothies cold-pressed juice acai bowls healthy drinks",
    "Ice Cream":      "gelato soft serve sundaes sorbet frozen desserts waffle cone",
    "Sandwiches":     "subs hoagies paninis wraps deli club sandwich hero",
    "Chicken":        "fried chicken wings rotisserie grilled chicken nuggets",
    "Hamburgers":     "burgers smash burger cheeseburger double patty gourmet",
    "Caribbean":      "jerk chicken rice peas plantains oxtail roti callaloo",
    "Ethiopian":      "injera berbere lentils stew communal eating East African",
    "Greek":          "souvlaki moussaka spanakopita tzatziki dolmades feta",
}


def stars(rating):
    if not rating:
        return ""
    full  = int(float(rating))
    half  = 1 if (float(rating) - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty


def price_label(tier):
    return {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}.get(tier, "")


def lexical_score(query, text):
    query_tokens = {token for token in str(query).lower().split() if len(token) > 2}
    text_tokens = {token for token in str(text).lower().split() if len(token) > 2}
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def build_photo_url(photo_ref, max_width=400):
    return f"{PLACES_PHOTO}?maxwidth={max_width}&photo_reference={photo_ref}&key={GOOGLE_API_KEY}"


@st.cache_data(ttl=86400, show_spinner=False)
def load_nyc_base(limit=8000):
    params = {
        "$limit": limit,
        "$where": "grade IN('A','B','C') AND cuisine_description IS NOT NULL AND dba IS NOT NULL",
        "$select": "camis,dba,boro,building,street,zipcode,cuisine_description,grade,score,latitude,longitude",
        "$order": "grade ASC",
    }
    try:
        r = requests.get(NYC_DOHMH_API, params=params, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
    except Exception as e:
        st.error(f"NYC Open Data fetch failed: {e}")
        return pd.DataFrame()

    df = df.drop_duplicates(subset=["camis"], keep="first")
    df["dba"]     = df["dba"].str.title().str.strip()
    df["cuisine"] = df["cuisine_description"].str.strip()
    df["boro"]    = df["boro"].str.title().str.strip()
    df["address"] = (
        df.get("building", pd.Series([""] * len(df))).fillna("") + " " +
        df.get("street",   pd.Series([""] * len(df))).fillna("") + ", " +
        df.get("boro",     pd.Series([""] * len(df))).fillna("") + ", NY " +
        df.get("zipcode",  pd.Series([""] * len(df))).fillna("")
    ).str.strip(", ")
    df["grade"] = df.get("grade", pd.Series(["N/A"] * len(df))).fillna("N/A")
    df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0).astype(int)
    df["lat"]   = pd.to_numeric(df.get("latitude",  None), errors="coerce")
    df["lon"]   = pd.to_numeric(df.get("longitude", None), errors="coerce")
    return df.reset_index(drop=True)


def fetch_google_place(name, address):
    query = f"{name} restaurant {address}"
    try:
        r = requests.get(PLACES_SEARCH, params={
            "query": query,
            "key": GOOGLE_API_KEY,
            "type": "restaurant",
            "region": "us",
        }, timeout=10)
        data = r.json()
        results = data.get("results", [])
        if not results:
            return None
        place_id = results[0].get("place_id")
        if not place_id:
            return None

        d = requests.get(PLACES_DETAILS, params={
            "place_id": place_id,
            "key": GOOGLE_API_KEY,
            "fields": "name,rating,user_ratings_total,price_level,editorial_summary,photos,url,opening_hours",
        }, timeout=10)
        det = d.json().get("result", {})

        rating    = det.get("rating")
        photos    = det.get("photos", [])
        photo_ref = photos[0].get("photo_reference") if photos else None

        if not photo_ref or not rating:
            return None

        return {
            "g_rating":    rating,
            "g_reviews":   det.get("user_ratings_total"),
            "g_price":     det.get("price_level"),
            "g_summary":   det.get("editorial_summary", {}).get("overview", ""),
            "g_photo_ref": photo_ref,
            "g_maps_url":  det.get("url", ""),
            "g_place_id":  place_id,
        }
    except Exception:
        return None


def enrich_with_google(nyc_df, sample_size):
    sample = nyc_df.sample(min(sample_size, len(nyc_df)), random_state=42).copy()
    enriched_rows = []
    progress = st.progress(0, text="Fetching Google Places data...")
    total = len(sample)

    for i, (_, row) in enumerate(sample.iterrows()):
        g = fetch_google_place(row["dba"], row["address"])
        if g:
            merged = row.to_dict()
            merged.update(g)
            enriched_rows.append(merged)
        time.sleep(0.12)
        if i % 10 == 0 or i == total - 1:
            pct  = min(int(((i + 1) / total) * 100), 99)
            kept = len(enriched_rows)
            progress.progress(pct, text=f"Google Places... {i+1}/{total} checked · {kept} enriched")

    progress.progress(100, text=f"Done — {len(enriched_rows)} restaurants indexed")
    time.sleep(0.4)
    progress.empty()

    if not enriched_rows:
        return pd.DataFrame()
    return pd.DataFrame(enriched_rows).reset_index(drop=True)


def load_or_enrich(base_df, sample_size):
    os.makedirs("data", exist_ok=True)

    if os.path.exists(ENRICHED_CACHE_PATH):
        age_hours = (time.time() - os.path.getmtime(ENRICHED_CACHE_PATH)) / 3600
        if age_hours < 168:
            try:
                df = pd.read_parquet(ENRICHED_CACHE_PATH)
                if not df.empty:
                    return df
            except Exception:
                pass

    df = enrich_with_google(base_df, sample_size)
    if not df.empty:
        df.to_parquet(ENRICHED_CACHE_PATH, index=False)
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


def build_description(row):
    name    = row.get("dba", "")
    cuisine = row.get("cuisine", "")
    boro    = row.get("boro", "")
    grade   = row.get("grade", "")
    score   = row.get("score", 0)
    address = row.get("address", "")
    extras  = CUISINE_KEYWORDS.get(cuisine, cuisine)
    summary = row.get("g_summary", "")
    rating  = row.get("g_rating")
    price   = row.get("g_price")
    reviews = row.get("g_reviews", 0)
    hygiene = {"A": "excellent hygiene", "B": "good hygiene", "C": "acceptable hygiene"}.get(grade, "")
    rating_str  = f"Google rating {rating}/5 based on {int(reviews):,} reviews." if rating and reviews else ""
    price_str   = f"Price level: {price_label(price)}." if price else ""
    summary_str = summary if summary else ""
    return (
        f"{name} is a {cuisine} restaurant in {boro}, New York City. "
        f"It serves {extras}. {summary_str} "
        f"{rating_str} {price_str} "
        f"Health inspection grade: {grade} ({hygiene}, score {score}). "
        f"Address: {address}."
    ).strip()


@st.cache_data(ttl=86400, show_spinner=False)
def compute_embeddings(_df):
    model = load_model()
    if model is None:
        return None
    return model.encode(
        _df["description"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def semantic_search(query, df, embeddings, top_k, boro_filter, grade_filter, min_rating):
    model = load_model()
    if model is not None and embeddings is not None:
        q_emb = model.encode([query], normalize_embeddings=True)
        scores = cosine_similarity(q_emb, embeddings)[0]
    else:
        scores = df["description"].fillna("").apply(lambda text: lexical_score(query, text)).to_numpy()

    mask = np.ones(len(df), dtype=bool)
    if boro_filter != "All":
        mask &= df["boro"].str.lower() == boro_filter.lower()
    if grade_filter != "All":
        mask &= df["grade"] == grade_filter
    if min_rating > 0:
        mask &= pd.to_numeric(df.get("g_rating", 0), errors="coerce").fillna(0) >= min_rating

    filtered = np.where(mask, scores, -1.0)
    top_idx  = np.argsort(filtered)[::-1][:top_k]
    results  = df.iloc[top_idx].copy()
    results["similarity"] = filtered[top_idx]
    return results[results["similarity"] > 0].reset_index(drop=True)


def render_card(row, rank):
    pct       = int(row["similarity"] * 100)
    grade     = row.get("grade", "N/A")
    rating    = row.get("g_rating")
    reviews   = row.get("g_reviews")
    price     = row.get("g_price")
    maps_url  = row.get("g_maps_url", "")
    photo_ref = row.get("g_photo_ref", "")
    cuisine   = row.get("cuisine", "")
    boro      = row.get("boro", "")
    address   = row.get("address", "")
    name      = row.get("dba", "Unknown")

    col_img, col_info = st.columns([1, 3])
    with col_img:
        if photo_ref:
            st.image(build_photo_url(photo_ref), use_container_width=True)
        else:
            st.markdown("🍽️")

    with col_info:
        grade_color = {"A": "🟢", "B": "🟡", "C": "🟠"}.get(grade, "⚪")
        st.markdown(f"### {name} {grade_color} Grade {grade}")
        st.caption(f"📍 {address}")
        if rating:
            rev_text = f"({int(reviews):,} reviews)" if reviews else ""
            st.markdown(f"{stars(rating)} **{float(rating):.1f}/5** {rev_text}")
        tags = [t for t in [cuisine, boro, price_label(price)] if t]
        if tags:
            st.markdown(" · ".join(f"`{t}`" for t in tags))
        desc = row.get("description", "")
        if len(desc) > 300:
            desc = desc[:297] + "..."
        st.write(desc)
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.metric("Match", f"{pct}%")
        with col_b:
            if maps_url:
                st.markdown(f"[📍 Open in Google Maps]({maps_url})")

    st.divider()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🍽️ NYC Restaurants")
    st.markdown("---")
    st.markdown("### Search Filters")

    sample_size  = st.slider("Restaurants to index", 100, 1000, 300, 50)
    boro_filter  = st.selectbox("Borough", ["All", "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
    grade_filter = st.selectbox("Health Grade", ["All", "A", "B", "C"])
    min_rating   = st.slider("Min Google Rating", 0.0, 5.0, 3.5, 0.5)
    top_k        = st.slider("Results to show", 3, 20, 8)

    st.markdown("---")
    if st.button("🔄 Refresh Restaurant Data"):
        if os.path.exists(ENRICHED_CACHE_PATH):
            os.remove(ENRICHED_CACHE_PATH)
        st.cache_data.clear()
        st.session_state["clustered_df"] = None
        st.session_state["raw_df"]       = None
        st.rerun()

    cache_status = "✅ Loaded from disk" if os.path.exists(ENRICHED_CACHE_PATH) else "🌐 Will fetch from Google"
    st.caption(cache_status)

    st.markdown("---")
    st.markdown("**Pages**")
    st.markdown("- 🔍 Search (this page)")
    st.markdown("- 📍 GIS Cluster Map")
    st.markdown("- 📊 PCA Embedding Explorer")
    st.markdown("- 🔮 Recommendations")


# ── Main ─────────────────────────────────────────────────────────────────────
st.title("🍽️ NYC Restaurant Semantic Search")
st.markdown("Search using natural language — describe a craving, vibe, or occasion.")
st.markdown("---")

with st.spinner("Loading NYC restaurant base data..."):
    base_df = load_nyc_base(limit=8000)

if base_df.empty:
    st.error("Failed to load NYC Open Data.")
    st.stop()

if "raw_df" not in st.session_state or st.session_state["raw_df"] is None:
    with st.spinner("Loading restaurant data (fetching from Google Places on first run, instant after)..."):
        enriched_df = load_or_enrich(base_df, sample_size)

    if enriched_df.empty:
        st.error("No restaurants returned valid Google data. Check your API key.")
        st.stop()

    enriched_df["description"]   = enriched_df.apply(build_description, axis=1)
    enriched_df["restaurant_id"] = enriched_df["camis"].astype(str)
    enriched_df["name"]          = enriched_df["dba"]
    enriched_df["lng"]           = enriched_df["lon"]
    enriched_df["cuisine_type"]  = enriched_df["cuisine"]
    enriched_df["price_tier"]    = pd.to_numeric(enriched_df.get("g_price", 2), errors="coerce").fillna(2).astype(int)
    enriched_df["avg_rating"]    = pd.to_numeric(enriched_df.get("g_rating", 3.0), errors="coerce").fillna(3.0)
    enriched_df["review_count"]  = pd.to_numeric(enriched_df.get("g_reviews", 0), errors="coerce").fillna(0).astype(int)

    st.session_state["raw_df"] = enriched_df
else:
    enriched_df = st.session_state["raw_df"]
    if "description" not in enriched_df.columns:
        enriched_df["description"] = enriched_df.apply(build_description, axis=1)
        st.session_state["raw_df"] = enriched_df

with st.spinner("Computing 384-dimensional embeddings..."):
    embeddings = compute_embeddings(enriched_df)

avg_r = pd.to_numeric(enriched_df["g_rating"], errors="coerce").dropna().mean()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Restaurants", f"{len(enriched_df):,}")
col2.metric("Embedding Dims", "384")
col3.metric("Avg Rating", f"{avg_r:.2f}★" if not pd.isna(avg_r) else "—")
col4.metric("Cuisine Types", enriched_df["cuisine"].nunique())
col5.metric("Boroughs", enriched_df["boro"].nunique())
st.markdown("---")

# Cluster filter integration
if st.session_state.get("clustered_df") is not None:
    cdf = st.session_state["clustered_df"]
    cluster_options = ["All Clusters"] + sorted(cdf["cluster_label"].unique().tolist())
    selected_cluster = st.sidebar.selectbox("🔮 Filter by Cluster", cluster_options)
    if selected_cluster != "All Clusters":
        valid_ids   = cdf[cdf["cluster_label"] == selected_cluster]["restaurant_id"].tolist()
        enriched_df = enriched_df[enriched_df["camis"].astype(str).isin(valid_ids)]

EXAMPLES = [
    "cozy Italian pasta spot in Brooklyn",
    "late night ramen and dumplings Manhattan",
    "healthy vegan grain bowls",
    "romantic French bistro with good wine",
    "authentic Mexican street tacos Queens",
    "spicy Korean BBQ grill",
    "fresh sushi omakase experience",
    "Sunday brunch with mimosas",
    "cheap and delicious Caribbean food Bronx",
    "upscale seafood and oyster bar",
]

col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input("What are you looking for?", placeholder="e.g. cozy Italian wine bar Brooklyn")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    go = st.button("Search", use_container_width=True)

ex = st.selectbox("Or try an example:", ["— choose an example —"] + EXAMPLES)
if ex != "— choose an example —":
    query = ex

st.markdown("---")

if query:
    with st.spinner(f"Searching {len(enriched_df):,} restaurants..."):
        results = semantic_search(query, enriched_df, embeddings, top_k, boro_filter, grade_filter, min_rating)
    if results.empty:
        st.info("No results found. Try adjusting filters or rephrasing.")
    else:
        st.success(f"**{len(results)} results** for *\"{query}\"*")
        for i, (_, row) in enumerate(results.iterrows()):
            render_card(row.to_dict(), i + 1)
else:
    st.info("Type a query above or pick an example to get started.")
