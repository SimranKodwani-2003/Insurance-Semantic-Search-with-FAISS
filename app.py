import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load model, index, metadata
@st.cache_resource
def load_faiss():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("insurance_faiss.index")
    with open("insurance_metadata.pkl", "rb") as f:
        documents = pickle.load(f)
    return model, index, documents

model, index, documents = load_faiss()

# Sidebar filters
st.sidebar.title("🔍 Filters")
sex_filter = st.sidebar.selectbox("Select Gender", ["All", "male", "female"])
smoker_filter = st.sidebar.selectbox("Smoker", ["All", "yes", "no"])
region_filter = st.sidebar.selectbox("Region", ["All", "northeast", "northwest", "southeast", "southwest"])

# Search input
st.title("🧠 Insurance Semantic Search with FAISS")
query = st.text_input("Enter your query:", "smoker with high charges and 2 children")

top_k = st.slider("Number of results:", 1, 50, 5)

def search_faiss_filtered(query, top_k, sex_filter, smoker_filter, region_filter):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k * 2)  # get more for filtering

    results = []
    for idx in indices[0]:
        row = pd.read_json(documents[idx], typ='series')
        if (
            (sex_filter == "All" or row['sex'] == sex_filter) and
            (smoker_filter == "All" or row['smoker'] == smoker_filter) and
            (region_filter == "All" or row['region'] == region_filter)
        ):
            results.append(row)

        if len(results) == top_k:
            break
    return results

# Search and display results
if st.button("Search"):
    matches = search_faiss_filtered(query, top_k, sex_filter, smoker_filter, region_filter)
    if matches:
        st.success(f"Top {len(matches)} matching records:")
        for i, row in enumerate(matches):
            st.markdown(f"### Result {i+1}")
            st.json(row.to_dict())
    else:
        st.warning("No matching records found.")
