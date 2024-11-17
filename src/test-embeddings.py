from embeddings import EmbeddingHandler
import json

def test_embedding_handler():
    try:
        # Initialize the handler
        print("Initializing EmbeddingHandler...")
        handler = EmbeddingHandler()
        
        # Test sample data
        sample_service = {
            'id': 'test1',
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
        
        print("\nStoring sample service...")
        handler.store_service(sample_service)
        
        print("\nTesting search...")
        query = "dog park with swimming pool"
        results = handler.search_similar(query)
        
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