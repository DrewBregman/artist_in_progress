import pinecone
import json
import random

# Initialize Pinecone (make sure your .env is loaded or set variables)
pinecone.init(api_key="PINECONE_API_KEY", environment="us-east1-gcp")
index = pinecone.Index("artist-complete-info-index")

# Create a simple dummy query embedding – ideally this should be generated using the same embedding method
# Here we use a random vector as a placeholder, but in practice use your Claude query embedding trick.
DIMENSION = 768
query_embedding = [random.uniform(-1, 1) for _ in range(DIMENSION)]

# Query the index for the top 3 matches
result = index.query(vector=query_embedding, top_k=3, include_metadata=True)
print(json.dumps(result.matches, indent=2))
