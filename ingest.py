
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

print("⏳ Loading CSV files...")
df_features = pd.read_csv('Bike_Features.csv', encoding='latin1')
df_reviews = pd.read_csv('Bikes_reviews.csv', encoding='utf-8-sig')

# 1. Initialize Persistent Client (Saves embeddings to disk)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_feature = chroma_client.get_or_create_collection(name="bike_features")
collection_reviews = chroma_client.get_or_create_collection(name="bike_reviews")

# 2. Build Text Strings
review_texts = [
    f"Review for {row['Varient_Name']}: {row['Review_title']} - {row['Review_description']} (Rating: {row['User_rating']}/5)"
    for _, row in df_reviews.iterrows()
]

def create_bike_description(row):
    return (
        f"The {row['Variant Name']} by {row['Company Name']} is a {row['Body Type']} "
        f"with an on-road price of {row['On-road prize']}. "
        f"Engine details: {row['Engine Type']}, {row['Displacement']} cc displacement, "
        f"producing {row['Peak Power']} power and {row['Max Torque']} torque. "
        f"It features a {row['No. of Cylinders']}-cylinder configuration with {row['Cooling System']} cooling, "
        f"{row['Fuel Supply']} fuel system, and {row['Gear Box']} transmission. "
        f"Performance metrics: City mileage of {row['City Mileage']}, Highway mileage of {row['Highway Mileage']}, "
        f"and 0-100 kmph acceleration in {row['0-100 Kmph (ec)']}."
    )

df_features["embedding_text"] = df_features.apply(create_bike_description, axis=1)
feature_texts = df_features["embedding_text"]

# 3. Compute Embeddings
print("⏳ Computing embeddings with SentenceTransformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
feature_embeddings = embedder.encode(feature_texts.tolist())
review_embeddings = embedder.encode(review_texts)

# 4. Populate Features Collection
print("💾 Saving Bike Features to ChromaDB...")
metadata_records = df_features[["Variant Name", "Company Name", "Body Type"]].fillna("N/A").to_dict(orient="records")
collection_feature.add(
    documents=feature_texts.tolist(),
    embeddings=feature_embeddings.tolist(),
    ids=[f"feature_{i}" for i in range(len(df_features))],
    metadatas=metadata_records
)

# 5. Populate Reviews Collection in Batches
print("💾 Saving Bike Reviews to ChromaDB...")
batch_size = 5000
for i in range(0, len(review_texts), batch_size):
    batch_review_texts = review_texts[i:i + batch_size]
    batch_review_embeddings = review_embeddings[i:i + batch_size].tolist()
    batch_ids = [f"review_{j}" for j in range(i, min(i + batch_size, len(df_reviews)))]
    batch_metadatas = df_reviews.iloc[i:i + batch_size][["Varient_Name", "User_rating"]].rename(columns={"Varient_Name": "Variant Name"}).to_dict(orient="records")

    collection_reviews.add(
        documents=batch_review_texts,
        embeddings=batch_review_embeddings,
        ids=batch_ids,
        metadatas=batch_metadatas
    )

print("✅ DONE! Persistent vector database saved to './chroma_db'")