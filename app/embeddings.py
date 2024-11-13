import os
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from pinecone import Pinecone, ServerlessSpec

# Initialize the tokenizer and model for generating embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Load a summarization pipeline from Hugging Face Transformers
summarizer = pipeline("summarization")

# Set up Pinecone client with error handling
api_key = os.getenv("PINECONE_API_KEY")
if api_key:
    try:
        pinecone_client = Pinecone(api_key=api_key)
        print("Pinecone client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Pinecone client: {e}")
        pinecone_client = None
else:
    print("PINECONE_API_KEY environment variable not set.")
    pinecone_client = None

index_name = "smbap"
dimension = 384  # Set the dimension to match the embedding model

# Check if index exists and delete if necessary, then recreate with correct dimensions
try:
    if index_name in [idx.name for idx in pinecone_client.list_indexes()]:
        pinecone_client.delete_index(index_name)  # Delete existing index first
    pinecone_client.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")  # Adjust region if necessary
    )
    print(f"Index '{index_name}' created successfully.")
    # Use the index with a corrected method for accessing it
    index = pinecone_client.Index(index_name)
except Exception as e:
    print(f"Error setting up index '{index_name}': {e}")
    index = None

def generate_embeddings(parsed_content):
    best_practices = parsed_content.get("best_practices", [])
    
    if isinstance(best_practices, list):
        input_text = "\n".join([str(item) for item in best_practices])
    else:
        input_text = str(best_practices)

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Store the embeddings in Pinecone only if the index is initialized
        if index:
            response = index.upsert([{
                'id': parsed_content.get('document_id', 'default_id'),  # Ensure you provide a unique ID for each document
                'values': embeddings.tolist()
            }])
            print(f"Embeddings upserted to Pinecone for document ID: {parsed_content.get('document_id', 'default_id')}")
        else:
            print("Index not available, skipping Pinecone upsert.")

        return embeddings.tolist()
    except Exception as e:
        print(f"Embedding Generation Error: {e}")
        return []

def generate_embeddings_and_store(parsed_content):
    """Wrapper function to generate embeddings and store them in Pinecone."""
    return generate_embeddings(parsed_content)

def generate_summary(parsed_content):
    best_practices = parsed_content.get("best_practices", [])
    input_text = " ".join(best_practices)

    # Maximum tokens that the model can handle
    max_input_tokens = 1024
    # Maximum tokens for each chunk of text for summarization
    chunk_size = 800

    # Split input text into chunks if it's too long
    input_words = input_text.split()
    summaries = []
    
    for i in range(0, len(input_words), chunk_size):
        chunk = " ".join(input_words[i:i + chunk_size])
        summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        summaries.append(summary)
    
    # Combine summaries of each chunk into a final summary
    final_summary = " ".join(summaries)
    
    return final_summary if summaries else "No content available for summarization."
