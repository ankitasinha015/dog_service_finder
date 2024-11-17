import streamlit as st
import json
from embeddings import EmbeddingHandler
from typing import Dict, Any
import os

class DogServiceApp:
    def __init__(self):
        self.embedding_handler = EmbeddingHandler()
        self.setup_streamlit()
        self.load_sample_data()

    def setup_streamlit(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="🐾 Paw Perfect",
            page_icon="🐕",
            layout="wide"
        )

    def load_sample_data(self):
        """Load and store sample data"""
        try:
            with open('data/sample_data.json', 'r') as f:
                data = json.load(f)
                for service in data['services']:
                    self.embedding_handler.store_service(service)
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")

    def render_search_box(self) -> str:
        """Render the search input box"""
        return st.text_input(
            "What kind of dog service are you looking for?",
            placeholder="Example: I need a dog daycare with training services"
        )

    def render_result(self, result: Dict[str, Any]):
        """Render a single result"""
        with st.container():
            st.markdown(f"""
            ### {result['metadata']['name']}
            ⭐ Rating: {result['metadata']['rating']}  
            📍 {result['metadata']['location']}  
            ⏰ {result['metadata']['hours']}  
            📞 {result['metadata']['contact']}
            """)
            
            st.markdown("**Services:**")
            for service in result['metadata']['services']:
                st.markdown(f"- {service}")
            
            st.markdown("**Features:**")
            for feature in result['metadata']['features']:
                st.markdown(f"- {feature}")
            
            st.markdown("**Recent Reviews:**")
            for review in result['metadata']['reviews'][:3]:
                st.markdown(f"> {review}")
            
            st.markdown("---")

    def run(self):
        """Main application loop"""
        st.title("🐾 Paw Perfect - Dog Service Finder")
        st.markdown("""
        Find the perfect services for your furry friend using natural language search.
        Simply describe what you're looking for!
        """)

        search_query = self.render_search_box()

        if search_query:
            with st.spinner("Searching for matches..."):
                results = self.embedding_handler.search_similar(search_query)
                
                if results:
                    st.success(f"Found {len(results)} matches!")
                    for result in results:
                        self.render_result(result)
                else:
                    st.warning("No matches found. Try adjusting your search terms.")

if __name__ == "__main__":
    app = DogServiceApp()
    app.run()
