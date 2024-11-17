import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import pinecone
from dotenv import load_dotenv

class EmbeddingHandler:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize the embedding model
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Initialize Pinecone
        self._init_pinecone()
    
    def _init_pinecone(self):
        """Initialize Pinecone connection"""
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        self.index = pinecone.Index(os.getenv("PINECONE_INDEX"))
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a given text"""
        return self.model.encode(text).tolist()
    
    def store_service(self, service: Dict[str, Any]):
        """Store a service in the vector database"""
        # Create text representation for embedding
        text_for_embedding = f"""
        {service['name']}. {service['description']}
        Services: {', '.join(service['services'])}
        Features: {', '.join(service['features'])}
        Reviews: {' '.join(service['reviews'])}
        """
        
        # Generate embedding
        embedding = self.create_embedding(text_for_embedding)
        
        # Store in Pinecone
        self.index.upsert(
            vectors=[(
                str(service['id']), 
                embedding,
                {
                    'name': service['name'],
                    'location': service['location'],
                    'services': service['services'],
                    'features': service['features'],
                    'rating': service['rating'],
                    'reviews': service['reviews'],
                    'contact': service['contact'],
                    'hours': service['hours']
                }
            )]
        )
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar services"""
        query_embedding = self.create_embedding(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results['matches']
