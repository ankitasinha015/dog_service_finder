import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingHandler:
    def __init__(self):
        """Initialize the EmbeddingHandler with environment setup and model loading"""
        self._load_environment()
        logger.info("Loading sentence transformer model...")
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self._init_pinecone()

    def _load_environment(self):
        """Load and validate environment variables"""
        try:
            # Get path to .env file
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            env_path = project_root / '.env'
            
            if not env_path.exists():
                raise FileNotFoundError(f".env file not found at {env_path}")
            
            load_dotenv(dotenv_path=env_path)
            
            # Verify required environment variables
            self.api_key = os.getenv('PINECONE_API_KEY')
            self.index_name = os.getenv('PINECONE_INDEX')
            
            if not self.api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
            if not self.index_name:
                raise ValueError("PINECONE_INDEX not found in environment variables")
                
            logger.info("Environment variables loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading environment variables: {str(e)}")
            raise

    def _init_pinecone(self):
        """Initialize Pinecone client and connect to index"""
        try:
            # Initialize Pinecone with your API key
            pc = Pinecone(api_key=self.api_key)
            
            # Connect to your index - using the specifications from your screenshot
            self.index = pc.Index(
                name=self.index_name,
                host="https://dogservice-91rn2hz.svc.aped-4627-b74a.pinecone.io"
            )
            
            logger.info(f"Successfully connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {str(e)}")
            raise

    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for the given text"""
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise

    def store_service(self, service: Dict[str, Any]):
        """Store a service in the vector database"""
        try:
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
            logger.info(f"Successfully stored service: {service['name']}")
            
        except Exception as e:
            logger.error(f"Error storing service: {str(e)}")
            raise

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar services based on query"""
        try:
            # Create query embedding
            query_embedding = self.create_embedding(query)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            logger.info(f"Found {len(results['matches'])} matches for query: {query}")
            return results['matches']
            
        except Exception as e:
            logger.error(f"Error searching services: {str(e)}")
            raise

    def delete_service(self, service_id: str):
        """Delete a service from the vector database"""
        try:
            self.index.delete(ids=[service_id])
            logger.info(f"Successfully deleted service with ID: {service_id}")
        except Exception as e:
            logger.error(f"Error deleting service: {str(e)}")
            raise

    def list_services(self) -> List[Dict]:
        """List all services in the database"""
        try:
            # Fetch all vectors (limit to 100 for safety)
            results = self.index.query(
                vector=[0] * 1536,  # Zero vector to fetch all
                top_k=100,
                include_metadata=True
            )
            return results['matches']
        except Exception as e:
            logger.error(f"Error listing services: {str(e)}")
            raise

# Add a test function to verify the setup
def test_embedding_handler():
    try:
        print("Testing EmbeddingHandler initialization...")
        handler = EmbeddingHandler()
        
        # Test sample service
        sample_service = {
            'id': 'test123',
            'name': 'Happy Paws Dog Park',
            'description': 'A premium dog park with training facilities',
            'services': ['Training', 'Day Care'],
            'features': ['Indoor Play Area', 'Swimming Pool'],
            'location': '123 Dog Lane',
            'rating': 4.8,
            'reviews': ['Great service!', 'Amazing facilities'],
            'contact': '555-0123',
            'hours': '9am-5pm'
        }
        
        # Store the service
        print("\nTesting service storage...")
        handler.store_service(sample_service)
        
        # Search for similar services
        print("\nTesting search functionality...")
        results = handler.search_similar("dog park with swimming pool")
        
        print("\nSearch Results:")
        for idx, result in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(f"Score: {result['score']}")
            print(f"Name: {result['metadata']['name']}")
            print(f"Location: {result['metadata']['location']}")
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")

if __name__ == "__main__":
    test_embedding_handler()