import os
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, INDEX_NAME

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists and retrieve it
try:
    if INDEX_NAME not in [idx.name for idx in pinecone_client.list_indexes()]:
        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=384,  # Set the dimension to match the embedding model
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region=PINECONE_ENVIRONMENT  # Replace with your specific environment
            )
        )

    # Access the index
    index = pinecone_client.Index(INDEX_NAME)
except Exception as e:
    print(f"Error accessing index '{INDEX_NAME}': {e}")
    index = None

def insert_embeddings(embedding_data):
    """Insert embeddings into Pinecone index."""
    if index is None:
        print("Index is not available.")
        return

    to_upsert = [(str(i), data["embedding"]) for i, data in enumerate(embedding_data)]
    try:
        index.upsert(vectors=to_upsert)
        print(f"Successfully inserted {len(to_upsert)} vectors into Pinecone.")
    except Exception as e:
        print(f"Error inserting embeddings into Pinecone: {e}")
